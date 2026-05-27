"""Thin typed HTTP client for the LumenX admin + public API.

All LumenX traffic goes through this module. Centralises auth header injection,
retry with exponential backoff on 5xx, timeouts, and token-masked logging.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from agent.config import settings

logger = logging.getLogger(__name__)

_ADMIN_PREFIX = "/api/admin"
_TIMEOUT = httpx.Timeout(10.0)
_RETRYABLE_STATUS = {500, 502, 503, 504}
_MAX_ATTEMPTS = 3


def _masked_token(token: str) -> str:
    return token[:4] + "…" if len(token) > 4 else "…"


class LumenXClient:
    """Sync client. Use as a context manager: `with LumenXClient() as c: ...`."""

    def __init__(
        self,
        base_url: str | None = None,
        admin_token: str | None = None,
    ) -> None:
        self._base_url = (base_url or settings.LUMENX_BASE_URL).rstrip("/")
        self._token = admin_token or settings.LUMENX_ADMIN_TOKEN
        self._client = httpx.Client(timeout=_TIMEOUT)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LumenXClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        url = f"{self._base_url}{path}"
        headers: dict[str, str] = dict(kw.pop("headers", {}) or {})
        if path.startswith(_ADMIN_PREFIX):
            headers["X-Admin-Token"] = self._token

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = self._client.request(method, url, headers=headers, **kw)
            except httpx.HTTPError as e:
                last_exc = e
                logger.warning(
                    "LumenX %s %s attempt %d/%d transport error: %s",
                    method, path, attempt, _MAX_ATTEMPTS, e,
                )
                if attempt == _MAX_ATTEMPTS:
                    raise
            else:
                if (
                    resp.status_code in _RETRYABLE_STATUS
                    and attempt < _MAX_ATTEMPTS
                ):
                    logger.warning(
                        "LumenX %s %s attempt %d/%d -> %d, retrying",
                        method, path, attempt, _MAX_ATTEMPTS, resp.status_code,
                    )
                else:
                    if resp.status_code >= 400:
                        logger.error(
                            "LumenX %s %s -> %d body=%s",
                            method, path, resp.status_code, resp.text[:300],
                        )
                    resp.raise_for_status()
                    logger.info(
                        "LumenX %s %s -> %d (token=%s)",
                        method, path, resp.status_code, _masked_token(self._token),
                    )
                    return resp.json() if resp.content else None

            time.sleep(0.5 * (2 ** (attempt - 1)))

        if last_exc:
            raise last_exc
        raise RuntimeError("LumenX request loop exited without result")

    # ---- admin endpoints ----
    def get_stats(self) -> dict:
        return self._request("GET", "/api/admin/stats")

    def get_inbox(self, since: str | None = None) -> dict:
        params = {"since": since} if since else None
        return self._request("GET", "/api/admin/inbox", params=params)

    def get_threads(self) -> list[dict] | dict:
        return self._request("GET", "/api/admin/threads")

    def get_thread(self, thread_id: str) -> dict:
        return self._request("GET", f"/api/admin/threads/{thread_id}")

    def post_reply(
        self,
        thread_id: str,
        text: str,
        draft_source: str | None = None,
        confidence: float | None = None,
    ) -> dict:
        body: dict[str, Any] = {"text": text}
        if draft_source is not None:
            body["draft_source"] = draft_source
        if confidence is not None:
            body["confidence"] = confidence
        return self._request("POST", f"/api/admin/threads/{thread_id}/reply", json=body)

    def mark_read(self, thread_id: str) -> dict:
        return self._request("POST", f"/api/admin/threads/{thread_id}/mark-read")

    def get_export(self) -> dict:
        return self._request("GET", "/api/admin/export")

    def get_products(self) -> dict | list[dict]:
        return self._request("GET", "/api/admin/products")

    def get_product(self, product_id: str) -> dict:
        return self._request("GET", f"/api/admin/products/{product_id}")
