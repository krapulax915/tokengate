"""Error types mapped to OpenAI-style HTTP error responses."""
from __future__ import annotations


class DietGateError(Exception):
    """Base class; `status` and OpenAI-style error fields."""

    status: int = 500
    err_type: str = "internal_error"
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadRequestError(DietGateError):
    status = 400
    err_type = "invalid_request_error"
    code = "bad_request"


class AuthError(DietGateError):
    status = 401
    err_type = "invalid_request_error"
    code = "invalid_api_key"


class RateLimitError(DietGateError):
    status = 429
    err_type = "rate_limit_error"
    code = "rate_limit_exceeded"


class SpendCapError(DietGateError):
    status = 429
    err_type = "invalid_request_error"
    code = "daily_spend_cap_reached"


class NotFoundError(DietGateError):
    status = 404
    err_type = "invalid_request_error"
    code = "not_found"


class UpstreamError(DietGateError):
    status = 502
    err_type = "upstream_error"
    code = "upstream_failure"


def error_body(err: DietGateError) -> dict:
    return {"error": {"message": err.message, "type": err.err_type, "code": err.code}}
