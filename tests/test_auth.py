"""Auth: 401 without/with wrong key, 429 rate limit, daily spend cap."""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from dietgate.core.auth import Authenticator
from dietgate.core.errors import AuthError, RateLimitError, SpendCapError
from tests.conftest import CHAT_URL, chat_body, runtime_of


async def test_missing_key_is_401(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(), headers={"Authorization": ""})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


async def test_wrong_key_is_401(client: AsyncClient) -> None:
    resp = await client.post(
        CHAT_URL, json=chat_body(), headers={"Authorization": "Bearer nope"}
    )
    assert resp.status_code == 401


async def test_good_key_passes(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body())
    assert resp.status_code == 200


def test_token_bucket_returns_429() -> None:
    authn = Authenticator(["k1"], requests_per_min=60, burst=2, daily_spend_cap_usd=10.0)
    masked = authn.check("Bearer k1")
    authn.check("Bearer k1")
    with pytest.raises(RateLimitError):
        authn.check("Bearer k1")  # burst exhausted, refill ~1 token/s


def test_spend_cap_returns_429() -> None:
    authn = Authenticator(["k1"], requests_per_min=6000, burst=10, daily_spend_cap_usd=0.05)
    masked = authn.check("Bearer k1")
    authn.add_spend(masked, 0.06)
    with pytest.raises(SpendCapError):
        authn.check("Bearer k1")


def test_daily_counter_resets_on_day_rollover() -> None:
    authn = Authenticator(["k1"], requests_per_min=6000, burst=10, daily_spend_cap_usd=0.05)
    masked = authn.check("Bearer k1")
    authn.add_spend(masked, 0.06)
    authn._day = "20000101"  # simulate midnight rollover (UTC day changed)
    assert authn.check("Bearer k1") == masked  # cap no longer hit


def test_key_never_stored_raw() -> None:
    authn = Authenticator(["super-secret"], requests_per_min=60, burst=10, daily_spend_cap_usd=1)
    masked = authn.check("Bearer super-secret")
    assert "super-secret" not in masked
    assert masked.startswith("key-")


async def test_rate_limit_integration_429(client: AsyncClient) -> None:
    rt = runtime_of(client)
    rt.authn._burst = 2.0
    rt.authn._refill_per_s = 0.001
    codes = []
    for _ in range(5):
        resp = await client.post(CHAT_URL, json=chat_body())
        codes.append(resp.status_code)
    assert codes[:2] == [200, 200]
    assert 429 in codes
