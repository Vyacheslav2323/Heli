"""Low-latency stream proxy for live video playback (HLS-aware)."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterator
from urllib.parse import urljoin

logger = logging.getLogger("perception.live_stream")

DEFAULT_YOUTUBE_LIVE = "https://www.youtube.com/watch?v=J7ZrIDvqlic"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


@dataclass
class StreamMeta:
    source: str
    resolved_url: str
    resolved_at_s: float
    is_hls: bool
    http_headers: dict[str, str]


class LiveStreamProxy:
    """Resolves and proxies a live stream URL without running CV."""

    def __init__(self, source: str = DEFAULT_YOUTUBE_LIVE, max_url_age_s: float = 240.0) -> None:
        self.source = source
        self.max_url_age_s = max_url_age_s
        self._lock = threading.Lock()
        self._meta: StreamMeta | None = None
        # Short tokens for huge googlevideo URLs (query-string length breaks browsers/servers).
        self._url_tokens: dict[str, tuple[str, float]] = {}
        self._token_ttl_s = 600.0

    def get_meta(self) -> StreamMeta:
        with self._lock:
            stale = (
                self._meta is None
                or (time.time() - self._meta.resolved_at_s) > self.max_url_age_s
            )
        if stale:
            self._refresh(force=True)
        with self._lock:
            assert self._meta is not None
            return self._meta

    def remember_url(self, url: str) -> str:
        token = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
        with self._lock:
            self._url_tokens[token] = (url, time.time())
            self._prune_tokens_locked()
        return token

    def resolve_token(self, token: str) -> str | None:
        with self._lock:
            item = self._url_tokens.get(token)
            if item is None:
                return None
            url, created = item
            if (time.time() - created) > self._token_ttl_s:
                self._url_tokens.pop(token, None)
                return None
            return url

    def fetch_url(self, url: str, *, timeout: float = 20.0) -> tuple[bytes, str]:
        headers = {"User-Agent": _UA}
        meta = self._meta
        if meta and meta.http_headers:
            headers.update(meta.http_headers)
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            return response.read(), content_type

    def fetch_stream(self, url: str, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        headers = {"User-Agent": _UA}
        meta = self._meta
        if meta and meta.http_headers:
            headers.update(meta.http_headers)
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def rewrite_playlist(self, playlist_text: str, playlist_url: str) -> str:
        """Rewrite media/playlist URIs to short same-origin proxy tokens."""
        out: list[str] = []
        for raw in playlist_text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                if 'URI="' in line:

                    def _sub(match: re.Match[str]) -> str:
                        abs_url = urljoin(playlist_url, match.group(1))
                        token = self.remember_url(abs_url)
                        return f'URI="/video/p/{token}"'

                    line = re.sub(r'URI="([^"]+)"', _sub, line)
                out.append(line)
                continue
            abs_url = urljoin(playlist_url, line)
            token = self.remember_url(abs_url)
            out.append(f"/video/p/{token}")
        return "\n".join(out) + "\n"

    def stream_bytes(self, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        while True:
            meta = self.get_meta()
            try:
                yield from self.fetch_stream(meta.resolved_url, chunk_size=chunk_size)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Upstream stream failed; refreshing URL: %s", exc)
                self._refresh(force=True)
                time.sleep(0.2)

    def _prune_tokens_locked(self) -> None:
        now = time.time()
        dead = [k for k, (_, t0) in self._url_tokens.items() if (now - t0) > self._token_ttl_s]
        for key in dead:
            self._url_tokens.pop(key, None)
        # Hard cap to avoid unbounded growth on long-running live.
        if len(self._url_tokens) > 4000:
            oldest = sorted(self._url_tokens.items(), key=lambda kv: kv[1][1])[:1000]
            for key, _ in oldest:
                self._url_tokens.pop(key, None)

    def _refresh(self, force: bool = False) -> None:
        with self._lock:
            if (
                not force
                and self._meta is not None
                and (time.time() - self._meta.resolved_at_s) <= self.max_url_age_s
            ):
                return
        resolved, headers = _resolve_source(self.source)
        is_hls = ".m3u8" in resolved.lower() or "hls" in resolved.lower()
        with self._lock:
            self._meta = StreamMeta(
                source=self.source,
                resolved_url=resolved,
                resolved_at_s=time.time(),
                is_hls=is_hls,
                http_headers=headers,
            )
        logger.info("Resolved live stream URL for %s (hls=%s)", self.source, is_hls)


def _resolve_source(source: str) -> tuple[str, dict[str, str]]:
    text = source.strip()
    if _is_youtube_url(text):
        return _youtube_stream_url(text)
    return text, {"User-Agent": _UA}


def _is_youtube_url(url: str) -> bool:
    return bool(
        re.search(
            r"(youtube\.com|youtu\.be|youtube-nocookie\.com)",
            url,
            flags=re.IGNORECASE,
        )
    )


def _youtube_stream_url(url: str) -> tuple[str, dict[str, str]]:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required for YouTube sources. pip install yt-dlp") from exc

    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "95/96/93/best[height<=720]/best[height<=1080]/best",
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            raise RuntimeError(f"yt-dlp returned no info for {url}")
        headers = dict(info.get("http_headers") or {})
        if "User-Agent" not in headers:
            headers["User-Agent"] = _UA
        direct = info.get("url")
        if direct:
            return str(direct), headers
        formats = info.get("formats") or []
        for fmt in reversed(formats):
            u = fmt.get("url")
            if not u:
                continue
            if fmt.get("protocol") in {"m3u8", "m3u8_native"} or str(u).endswith(".m3u8"):
                fmt_headers = dict(fmt.get("http_headers") or headers)
                return str(u), fmt_headers
        for fmt in reversed(formats):
            if fmt.get("url") and fmt.get("vcodec") not in (None, "none"):
                fmt_headers = dict(fmt.get("http_headers") or headers)
                return str(fmt["url"]), fmt_headers
    raise RuntimeError(f"Could not resolve playable stream URL for {url}")


def is_playlist_payload(url: str, content_type: str, body: bytes) -> bool:
    """Detect HLS playlists. YouTube segment URLs often contain '.m3u8' mid-path."""
    head = body[:64].lstrip()
    if head.startswith(b"#EXTM3U"):
        return True
    ct = (content_type or "").lower()
    if "mpegurl" in ct:
        # googlevideo sometimes labels TS as mpegurl incorrectly; prefer body sniff.
        return head.startswith(b"#") or head.startswith(b"#EXT")
    path = url.split("?", 1)[0].lower()
    return path.endswith(".m3u8")
