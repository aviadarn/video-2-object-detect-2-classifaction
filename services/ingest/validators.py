from __future__ import annotations

from urllib.parse import urlparse


class ValidationError(ValueError):
    pass


def validate_video_url(video_url: str, allowed_hosts: tuple[str, ...]) -> None:
    parsed = urlparse(video_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"}:
        raise ValidationError("Only http/https URLs are allowed")

    if host not in allowed_hosts:
        raise ValidationError(
            f"Host '{host}' is not allowed. Allowed hosts: {', '.join(sorted(allowed_hosts))}"
        )
