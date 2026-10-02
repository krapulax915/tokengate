"""API key check, per-key token bucket, daily spend cap (spec section 5.2).

Pure in-memory, synchronous: runs on the hot path with no awaits.
"""
from __future__ import annotations

import hashlib
import time


class _Bucket:
    __slots__ = ("tokens", "updated_mono")

    def __init__(self, tokens: float, updated_mono: float) -> None:
        self.tokens = tokens
        self.updated_mono = updated_mono


class Authenticator:
    def __init__(
        self,
        api_keys: list[str],
        requests_per_min: float,
        burst: int,
        daily_spend_cap_usd: float,
    ) -> None:
        self._keys = {str(k) for k in api_keys}
        self._refill_per_s = max(0.0, requests_per_min / 60.0)
        self._burst = float(burst)
        self._cap_usd = daily_spend_cap_usd
        self._buckets: dict[str, _Bucket] = {}
        self._day_spend: dict[str, float] = {}
        self._day: str = time.strftime("%Y%m%d", time.gmtime())

    def check(self, authorization_header: str | None) -> str:
        """Return a masked key id or raise 401/429. `add_spend` is called later."""
        self._roll_day_if_needed()
        raw_key = self._extract_key(authorization_header)
        if raw_key is None or raw_key not in self._keys:
            from dietgate.core.errors import AuthError

            raise AuthError("missing or invalid API key")
        key_id = self._mask(raw_key)
        self._take_token(key_id)
        from dietgate.core.errors import SpendCapError

        if self._day_spend.get(key_id, 0.0) >= self._cap_usd:
            raise SpendCapError("daily spend cap reached for this API key")
        return key_id

    def add_spend(self, masked_key_id: str, usd: float) -> None:
        self._roll_day_if_needed()
        self._day_spend[masked_key_id] = self._day_spend.get(masked_key_id, 0.0) + usd

    def spent_today(self, masked_key_id: str) -> float:
        return self._day_spend.get(masked_key_id, 0.0)

    # ------------------------------------------------------------- internals
    @staticmethod
    def _extract_key(header: str | None) -> str | None:
        if not header:
            return None
        parts = header.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return header.strip() or None

    @staticmethod
    def _mask(key: str | None) -> str | None:
        """Keys are never stored or logged raw; only a hash prefix travels."""
        if not key:
            return None
        return "key-" + hashlib.sha256(key.encode()).hexdigest()[:10]

    def _take_token(self, key_id: str) -> None:
        from dietgate.core.errors import RateLimitError

        now = time.monotonic()
        bucket = self._buckets.get(key_id)
        if bucket is None:
            bucket = _Bucket(tokens=self._burst, updated_mono=now)
            self._buckets[key_id] = bucket
        refill = (now - bucket.updated_mono) * self._refill_per_s
        bucket.tokens = min(self._burst, bucket.tokens + refill)
        bucket.updated_mono = now
        if bucket.tokens < 1.0:
            raise RateLimitError("rate limit exceeded for this API key")
        bucket.tokens -= 1.0

    def _roll_day_if_needed(self) -> None:
        today = time.strftime("%Y%m%d", time.gmtime())
        if today != self._day:
            self._day = today
            self._day_spend.clear()
