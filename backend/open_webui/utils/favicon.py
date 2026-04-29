"""Server-side favicon proxy with SSRF protection, singleflight dedup, and caching."""

from __future__ import annotations

import asyncio
import hashlib
import html
import ipaddress
import logging
import socket
import time
from collections import OrderedDict
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from fastapi import Request, Response

from open_webui.env import REDIS_KEY_PREFIX

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_FAVICON_CACHE_TTL = 7 * 24 * 3600  # 7 days
_FAVICON_TIMEOUT = aiohttp.ClientTimeout(total=5)
_MAX_ICON_BYTES = 256 * 1024  # 256 KiB
_MAX_HTML_BYTES = 250 * 1024  # 250 KiB
_MAX_REDIRECTS = 3
_MAX_CACHE_BYTES = 50 * 1024 * 1024  # 50 MiB byte budget

_ALLOWED_ICON_TYPES = {
    'image/x-icon',
    'image/vnd.microsoft.icon',
    'image/png',
    'image/jpeg',
    'image/gif',
    'image/webp',
}

_BADGE_PALETTE = [
    '#e74c3c', '#e67e22', '#2ecc71', '#3498db',
    '#9b59b6', '#1abc9c', '#e91e63', '#f39c12',
]

_RESPONSE_HEADERS = {
    'Cache-Control': 'public, max-age=604800',
    'X-Content-Type-Options': 'nosniff',
}


# ─── Pure helpers ─────────────────────────────────────────────────────────────

def _svg_badge(hostname: str) -> bytes:
    """Generate a colored circle+letter SVG for a hostname."""
    if hostname:
        letter = html.escape(hostname[0].upper())
        idx = hashlib.sha256(hostname.encode()).digest()[0] % len(_BADGE_PALETTE)
        color = _BADGE_PALETTE[idx]
    else:
        letter = '?'
        color = '#95a5a6'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        f'<circle cx="16" cy="16" r="16" fill="{color}"/>'
        f'<text x="16" y="21" font-size="16" font-family="sans-serif" '
        f'fill="white" text-anchor="middle">{letter}</text>'
        f'</svg>'
    ).encode()


def normalize_hostname(url: str) -> str | None:
    """Extract and validate hostname from a URL string."""
    try:
        url = url.strip()
        if '://' not in url:
            url = 'https://' + url
        parsed = urlparse(url)
        hostname = (parsed.hostname or '').lower().rstrip('.')
        if not hostname:
            return None
        # Accept raw IPs directly
        try:
            ipaddress.ip_address(hostname)
            return hostname
        except ValueError:
            pass
        # Require at least one dot for real domains
        if '.' not in hostname:
            return None
        try:
            hostname = hostname.encode('idna').decode('ascii')
        except (UnicodeError, UnicodeDecodeError):
            return None
        return hostname or None
    except Exception:
        return None


def _is_safe_url(url: str) -> bool:
    """Return True if url has an http/https scheme and a globally routable target."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        try:
            if not ipaddress.ip_address(hostname).is_global:
                return False
        except ValueError:
            pass  # hostname; _SafeResolver handles it
        return True
    except Exception:
        return False


# ─── SSRF-safe networking ─────────────────────────────────────────────────────

class _SafeResolver(aiohttp.ThreadedResolver):
    """DNS resolver that rejects non-global IPs (prevents DNS rebinding)."""

    async def resolve(self, hostname, port=0, family=socket.AF_UNSPEC):
        results = await super().resolve(hostname, port, family)
        for r in results:
            if not ipaddress.ip_address(r['host']).is_global:
                raise OSError(f'non-global IP: {r["host"]}')
        return results


# ─── Shared session (lazy singleton) ─────────────────────────────────────────

_session: aiohttp.ClientSession | None = None


async def _get_session() -> aiohttp.ClientSession:
    """Lazily create and reuse a single aiohttp session with SSRF-safe resolver."""
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(resolver=_SafeResolver(), use_dns_cache=False)
        _session = aiohttp.ClientSession(
            connector=connector,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; open-webui/favicon-proxy)'},
        )
    return _session


# ─── Bounded fetch ────────────────────────────────────────────────────────────

async def _fetch_bounded(
    session: aiohttp.ClientSession,
    url: str,
    max_bytes: int,
    allowed_types: set[str] | None,
) -> tuple[bytes, str] | None:
    """GET with capped size and per-hop SSRF validation."""
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        try:
            async with session.get(
                current, allow_redirects=False, timeout=_FAVICON_TIMEOUT
            ) as resp:
                if resp.status in (301, 302, 303, 307, 308):
                    location = resp.headers.get('Location', '')
                    if not location:
                        return None
                    next_url = urljoin(current, location)
                    if not _is_safe_url(next_url):
                        return None
                    current = next_url
                    continue

                if resp.status != 200:
                    return None

                ct = resp.headers.get('Content-Type', '').split(';')[0].strip().lower()
                if allowed_types is not None and ct not in allowed_types:
                    return None

                data = b''
                async for chunk in resp.content.iter_chunked(8192):
                    data += chunk
                    if len(data) > max_bytes:
                        return None
                return data, ct
        except Exception:
            return None
    return None


# ─── Favicon discovery ────────────────────────────────────────────────────────

async def _discover_favicon_impl(hostname: str) -> tuple[bytes, str] | None:
    """Fetch favicon for hostname: try /favicon.ico, then parse HTML <link> tags."""
    # Guard against raw private IPs bypassing the resolver
    try:
        if not ipaddress.ip_address(hostname).is_global:
            return None
    except ValueError:
        pass  # hostname; _SafeResolver handles it

    session = await _get_session()

    # Try /favicon.ico first
    result = await _fetch_bounded(
        session, f'https://{hostname}/favicon.ico', _MAX_ICON_BYTES, _ALLOWED_ICON_TYPES
    )
    if result:
        return result

    # Fall back to HTML <link> discovery
    html_result = await _fetch_bounded(
        session, f'https://{hostname}/', _MAX_HTML_BYTES, {'text/html'}
    )
    if not html_result:
        return None

    html_bytes, _ = html_result
    try:
        soup = BeautifulSoup(html_bytes, 'html.parser')
    except Exception:
        return None

    candidates: list[tuple[int, int, str]] = []
    for tag in soup.find_all('link'):
        rel = tag.get('rel') or []
        if isinstance(rel, str):
            rel = [rel]
        rel_lower = [r.lower() for r in rel]

        href = tag.get('href') or ''
        if not href or href.startswith('data:'):
            continue
        tag_type = (tag.get('type') or '').lower()
        if 'svg' in tag_type:
            continue

        if 'apple-touch-icon' in rel_lower and 'precomposed' not in rel_lower:
            priority, size = 0, 180
        elif 'icon' in rel_lower:
            priority = 1
            size = 0
            sizes_str = (tag.get('sizes') or '').strip()
            if sizes_str and sizes_str.lower() != 'any':
                try:
                    size = max(
                        int(p.lower().split('x')[0])
                        for p in sizes_str.split()
                        if 'x' in p.lower()
                    )
                except Exception:
                    size = 0
        elif 'shortcut' in rel_lower:
            priority, size = 2, 0
        elif 'apple-touch-icon' in rel_lower:  # precomposed
            priority, size = 3, 0
        else:
            continue

        full_href = urljoin(f'https://{hostname}/', href)
        candidates.append((priority, -size, full_href))

    candidates.sort(key=lambda x: (x[0], x[1]))

    for _, _, href in candidates:
        if not _is_safe_url(href):
            continue
        icon_result = await _fetch_bounded(session, href, _MAX_ICON_BYTES, _ALLOWED_ICON_TYPES)
        if icon_result:
            return icon_result

    return None


# ─── Singleflight deduplication ───────────────────────────────────────────────

_inflight: dict[str, asyncio.Future] = {}


async def _discover_favicon(hostname: str) -> tuple[bytes, str] | None:
    """Coalesce concurrent requests for the same hostname into a single fetch."""
    if hostname in _inflight:
        try:
            return await _inflight[hostname]
        except Exception:
            return None  # Graceful fallback for waiters

    future = asyncio.get_event_loop().create_future()
    _inflight[hostname] = future
    try:
        result = await _discover_favicon_impl(hostname)
        future.set_result(result)
        return result
    except Exception as e:
        future.set_exception(e)
        return None  # Graceful fallback for leader
    finally:
        _inflight.pop(hostname, None)


# ─── Byte-bounded in-memory LRU cache ────────────────────────────────────────

_cache: OrderedDict = OrderedDict()
_cache_bytes: int = 0


def _cache_get(hostname: str) -> tuple[bytes, str] | None:
    global _cache_bytes
    entry = _cache.get(hostname)
    if entry is None:
        return None
    content, ct, expires_at = entry
    if time.monotonic() > expires_at:
        _cache_bytes -= len(content)
        del _cache[hostname]
        return None
    _cache.move_to_end(hostname)
    return content, ct


def _cache_set(hostname: str, content: bytes, ct: str) -> None:
    global _cache_bytes
    if hostname in _cache:
        old_content, _, _ = _cache[hostname]
        _cache_bytes -= len(old_content)
    _cache[hostname] = (content, ct, time.monotonic() + _FAVICON_CACHE_TTL)
    _cache_bytes += len(content)
    _cache.move_to_end(hostname)
    # Evict by byte budget
    while _cache_bytes > _MAX_CACHE_BYTES and _cache:
        _, (evicted_content, _, _) = _cache.popitem(last=False)
        _cache_bytes -= len(evicted_content)


# ─── Redis cache layer (multi-worker) ────────────────────────────────────────

async def _get_cached(redis, hostname: str) -> tuple[bytes, str] | None:
    """Return cached (content, ct) from Redis or in-memory."""
    if redis is not None:
        try:
            raw = await redis.get(f'{REDIS_KEY_PREFIX}:favicon:{hostname}')
            if raw:
                ct_bytes, _, content = raw.partition(b'\n')
                return content, ct_bytes.decode()
        except Exception:
            pass
        return None
    return _cache_get(hostname)


async def _set_cached(redis, hostname: str, content: bytes, ct: str) -> None:
    """Write (content, ct) to Redis (with TTL) or in-memory."""
    if redis is not None:
        try:
            await redis.set(
                f'{REDIS_KEY_PREFIX}:favicon:{hostname}',
                ct.encode() + b'\n' + content,
                ex=_FAVICON_CACHE_TTL,
            )
        except Exception:
            pass
    else:
        _cache_set(hostname, content, ct)


# ─── Main entry point ─────────────────────────────────────────────────────────

async def get_favicon(request: Request, url: str) -> Response:
    """Favicon proxy endpoint handler; always returns a valid image response."""
    hostname = normalize_hostname(url)
    redis = request.app.state.redis

    if hostname:
        cached = await _get_cached(redis, hostname)
        if cached:
            content, ct = cached
            return Response(content=content, media_type=ct, headers=_RESPONSE_HEADERS)

    icon: tuple[bytes, str] | None = None
    if hostname:
        try:
            icon = await _discover_favicon(hostname)
        except Exception:
            pass

    content, ct = icon if icon else (_svg_badge(hostname or ''), 'image/svg+xml')

    if hostname:
        await _set_cached(redis, hostname, content, ct)

    return Response(content=content, media_type=ct, headers=_RESPONSE_HEADERS)
