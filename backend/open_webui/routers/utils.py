import black
import hashlib
import html
import ipaddress
import logging
import markdown
import socket
import time
from collections import OrderedDict
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from open_webui.models.chats import ChatTitleMessagesForm
from open_webui.config import DATA_DIR, ENABLE_ADMIN_EXPORT
from open_webui.constants import ERROR_MESSAGES
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from starlette.responses import FileResponse


from open_webui.env import REDIS_KEY_PREFIX
from open_webui.utils.misc import get_gravatar_url
from open_webui.utils.pdf_generator import PDFGenerator
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.code_interpreter import execute_code_jupyter

log = logging.getLogger(__name__)

router = APIRouter()


@router.get('/gravatar')
async def get_gravatar(email: str, user=Depends(get_verified_user)):
    return get_gravatar_url(email)


class CodeForm(BaseModel):
    code: str


@router.post('/code/format')
async def format_code(form_data: CodeForm, user=Depends(get_admin_user)):
    try:
        formatted_code = black.format_str(form_data.code, mode=black.Mode())
        return {'code': formatted_code}
    except black.NothingChanged:
        return {'code': form_data.code}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post('/code/execute')
async def execute_code(request: Request, form_data: CodeForm, user=Depends(get_verified_user)):
    if not request.app.state.config.ENABLE_CODE_EXECUTION:
        raise HTTPException(
            status_code=403,
            detail=ERROR_MESSAGES.FEATURE_DISABLED('Code execution'),
        )

    if request.app.state.config.CODE_EXECUTION_ENGINE == 'jupyter':
        output = await execute_code_jupyter(
            request.app.state.config.CODE_EXECUTION_JUPYTER_URL,
            form_data.code,
            (
                request.app.state.config.CODE_EXECUTION_JUPYTER_AUTH_TOKEN
                if request.app.state.config.CODE_EXECUTION_JUPYTER_AUTH == 'token'
                else None
            ),
            (
                request.app.state.config.CODE_EXECUTION_JUPYTER_AUTH_PASSWORD
                if request.app.state.config.CODE_EXECUTION_JUPYTER_AUTH == 'password'
                else None
            ),
            request.app.state.config.CODE_EXECUTION_JUPYTER_TIMEOUT,
        )

        return output
    else:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.DEFAULT('Code execution engine not supported'),
        )


class MarkdownForm(BaseModel):
    md: str


@router.post('/markdown')
async def get_html_from_markdown(form_data: MarkdownForm, user=Depends(get_verified_user)):
    return {'html': markdown.markdown(form_data.md)}


class ChatForm(BaseModel):
    title: str
    messages: list[dict]


@router.post('/pdf')
async def download_chat_as_pdf(form_data: ChatTitleMessagesForm, user=Depends(get_verified_user)):
    try:
        pdf_bytes = PDFGenerator(form_data).generate_chat_pdf()

        return Response(
            content=pdf_bytes,
            media_type='application/pdf',
            headers={'Content-Disposition': 'attachment;filename=chat.pdf'},
        )
    except Exception as e:
        log.exception(f'Error generating PDF: {e}')
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/db/download')
async def download_db(user=Depends(get_admin_user)):
    if not ENABLE_ADMIN_EXPORT:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    from open_webui.internal.db import engine

    if engine.name != 'sqlite':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DB_NOT_SQLITE,
        )
    return FileResponse(
        engine.url.database,
        media_type='application/octet-stream',
        filename='webui.db',
    )


_FAVICON_CACHE: OrderedDict = OrderedDict()
_FAVICON_CACHE_MAX = 1000
_FAVICON_CACHE_TTL = 7 * 24 * 3600
_FAVICON_TIMEOUT = aiohttp.ClientTimeout(total=5)
_MAX_ICON_BYTES = 256 * 1024   # 256 KiB
_MAX_HTML_BYTES = 250 * 1024   # 250 KiB
_MAX_REDIRECTS = 3
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


def _svg_badge(hostname: str) -> bytes:
    if hostname:
        letter = html.escape(hostname[0].upper())
        idx = hashlib.sha256(hostname.encode()).digest()[0] % len(_BADGE_PALETTE)
        color = _BADGE_PALETTE[idx]
    else:
        letter = '?'
        color = '#95a5a6'
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        f'<circle cx="16" cy="16" r="16" fill="{color}"/>'
        f'<text x="16" y="21" font-size="16" font-family="sans-serif" '
        f'fill="white" text-anchor="middle">{letter}</text>'
        f'</svg>'
    )
    return svg.encode()


def _normalize_hostname(url: str) -> str | None:
    try:
        url = url.strip()
        if '://' not in url:
            url = 'https://' + url
        parsed = urlparse(url)
        hostname = (parsed.hostname or '').lower().rstrip('.')
        if not hostname:
            return None
        try:
            hostname = hostname.encode('idna').decode('ascii')
        except (UnicodeError, UnicodeDecodeError):
            pass
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
            if not ipaddress.ip_address(hostname).is_global:  # raw IP
                return False
        except ValueError:
            pass  # hostname; _SafeResolver handles it
        return True
    except Exception:
        return False


class _SafeResolver(aiohttp.ThreadedResolver):
    async def resolve(self, hostname, port=0, family=socket.AF_UNSPEC):
        results = await super().resolve(hostname, port, family)
        for r in results:
            if not ipaddress.ip_address(r['host']).is_global:
                raise OSError(f'non-global IP: {r["host"]}')
        return results


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


async def _discover_favicon(hostname: str) -> tuple[bytes, str] | None:
    # aiohttp bypasses the resolver for raw numeric IPs (is_ip_address() short-circuits
    # _resolve_host). This is the sole guard for that path.
    try:
        if not ipaddress.ip_address(hostname).is_global:
            return None
    except ValueError:
        pass  # hostname; _SafeResolver handles it

    connector = aiohttp.TCPConnector(resolver=_SafeResolver(), use_dns_cache=False)
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; open-webui/favicon-proxy)'}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        # try /favicon.ico
        result = await _fetch_bounded(
            session,
            f'https://{hostname}/favicon.ico',
            _MAX_ICON_BYTES,
            _ALLOWED_ICON_TYPES,
        )
        if result:
            return result

        # fall back to HTML <link> discovery
        html_result = await _fetch_bounded(
            session,
            f'https://{hostname}/',
            _MAX_HTML_BYTES,
            {'text/html'},
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
            icon_result = await _fetch_bounded(
                session, href, _MAX_ICON_BYTES, _ALLOWED_ICON_TYPES
            )
            if icon_result:
                return icon_result

    return None


def _cache_get(hostname: str) -> tuple[bytes, str] | None:
    entry = _FAVICON_CACHE.get(hostname)
    if entry is None:
        return None
    content, ct, expires_at = entry
    if time.monotonic() > expires_at:
        del _FAVICON_CACHE[hostname]
        return None
    _FAVICON_CACHE.move_to_end(hostname)
    return content, ct


def _cache_set(hostname: str, content: bytes, ct: str) -> None:
    if hostname in _FAVICON_CACHE:
        _FAVICON_CACHE.move_to_end(hostname)
    _FAVICON_CACHE[hostname] = (content, ct, time.monotonic() + _FAVICON_CACHE_TTL)
    while len(_FAVICON_CACHE) > _FAVICON_CACHE_MAX:
        _FAVICON_CACHE.popitem(last=False)


_FAVICON_RESPONSE_HEADERS = {
    'Cache-Control': 'public, max-age=604800',
    'X-Content-Type-Options': 'nosniff',
}


async def _get_cached_favicon(redis, hostname: str) -> tuple[bytes, str] | None:
    """Return cached (content, ct) from Redis or in-memory, whichever is configured."""
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


async def _set_cached_favicon(redis, hostname: str, content: bytes, ct: str) -> None:
    """Write (content, ct) to Redis (with TTL) or in-memory, whichever is configured."""
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


@router.get('/favicon')
async def get_favicon(request: Request, url: str = '', user=Depends(get_verified_user)):
    """Favicon proxy; always returns a response."""
    hostname = _normalize_hostname(url)
    redis = request.app.state.redis

    if hostname:
        cached = await _get_cached_favicon(redis, hostname)
        if cached:
            content, ct = cached
            return Response(content=content, media_type=ct, headers=_FAVICON_RESPONSE_HEADERS)

    icon: tuple[bytes, str] | None = None
    if hostname:
        try:
            icon = await _discover_favicon(hostname)
        except Exception:
            pass

    content, ct = icon if icon else (_svg_badge(hostname or ''), 'image/svg+xml')

    if hostname:
        await _set_cached_favicon(redis, hostname, content, ct)

    return Response(content=content, media_type=ct, headers=_FAVICON_RESPONSE_HEADERS)
