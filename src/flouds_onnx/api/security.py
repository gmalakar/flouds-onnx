# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
flouds_onnx.api.security
~~~~~~~~~~~~~~~~~~~~~~~~
Access control for the optional HTTP server.

The server can download and export models, load them into memory and delete them from
disk, so every route except ``/health`` is protected. There are two modes:

Token mode: ``FLOUDS_ONNX_API_TOKEN`` is set.
    Every request must send ``Authorization: Bearer <token>``. This is the only mode
    in which the server may listen on a non-loopback address.

Local mode: no token is set.
    Only requests from this machine are served: the connection must come from a
    loopback address and the ``Host`` header must name a loopback host. The second
    check stops DNS-rebinding, where a web page makes the browser reach
    ``127.0.0.1`` under the page's own host name.

In both modes a request that carries an ``Origin`` header (that is, one made by a web
page) is refused unless the origin is the server itself or is listed in
``FLOUDS_ONNX_ALLOWED_ORIGINS``. Without that, any page open in a browser on the same
machine could drive a server running in local mode.
"""
from __future__ import annotations

import hmac
import ipaddress
import os
from typing import FrozenSet, Optional
from urllib.parse import urlsplit

API_TOKEN_ENV = "FLOUDS_ONNX_API_TOKEN"  # nosec B105 - the name of a variable, not a secret
ALLOWED_ORIGINS_ENV = "FLOUDS_ONNX_ALLOWED_ORIGINS"

_LOOPBACK_NAMES = frozenset({"localhost"})


def configured_token() -> Optional[str]:
    token = (os.environ.get(API_TOKEN_ENV) or "").strip()
    return token or None


def configured_origins() -> FrozenSet[str]:
    raw = os.environ.get(ALLOWED_ORIGINS_ENV) or ""
    return frozenset(item.strip().rstrip("/").lower() for item in raw.split(",") if item.strip())


def is_loopback_host(host: Optional[str]) -> bool:
    """True for localhost, 127.0.0.0/8 and ::1. Accepts 'host', 'host:port' and '[::1]:port'."""
    if not host:
        return False
    value = host.strip().lower()
    if value.startswith("["):  # [::1]:19720
        value = value[1:].split("]", 1)[0]
    elif value.count(":") == 1:  # name:port or 127.0.0.1:port
        value = value.split(":", 1)[0]
    if value in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _origin_allowed(origin: str, host_header: str, allowed: FrozenSet[str]) -> bool:
    normalised = origin.strip().rstrip("/").lower()
    if normalised in allowed:
        return True
    # Same origin: the page was served by this server (for example the API docs page).
    return urlsplit(normalised).netloc == host_header.strip().lower() and bool(host_header.strip())


def check_access(
    *, token: Optional[str], allowed_origins: FrozenSet[str], authorization: str, origin: str, host_header: str, client_host: Optional[str]
) -> tuple[int, str]:
    """Decide whether a request may proceed. Returns (status, reason); status 200 means yes."""
    if origin and not _origin_allowed(origin, host_header, allowed_origins):
        return 403, "Requests from this web origin are not allowed."

    if token:
        scheme, _, supplied = authorization.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.strip().encode("utf-8"), token.encode("utf-8")):
            return 401, "A valid bearer token is required."
        return 200, ""

    if not is_loopback_host(client_host) or not is_loopback_host(host_header):
        return 403, f"This server has no API token configured and only accepts local requests. Set {API_TOKEN_ENV} to allow remote access."
    return 200, ""
