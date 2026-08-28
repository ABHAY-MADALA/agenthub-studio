"""
oauth.py
========
Small local-demo OAuth helper for Google Workspace integrations.

Tokens are stored in storage/tokens/, which is gitignored. This is fine for a
single-user local portfolio demo. A production version should encrypt tokens
and store them per user in a database or secret store.
"""

import json
import secrets
import ssl
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import certifi
import config

logger = config.get_logger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

GOOGLE_TOOL_KEYS = {"gmail", "google_drive", "google_calendar"}
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class OAuthConfigError(RuntimeError):
    """Raised when the app is missing OAuth client credentials."""


def _token_path(provider: str) -> Path:
    return config.TOKENS_DIR / f"{provider}_token.json"


def _state_path(provider: str) -> Path:
    return config.TOKENS_DIR / f"{provider}_state.json"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read OAuth file: %s", path)
        return None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _post_form(url: str, data: Dict[str, str], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    body = urlencode(data).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
    with urlopen(req, timeout=20, context=SSL_CONTEXT) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, access_token: str, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "User-Agent": "AgentHub-Studio",
            **(extra_headers or {}),
        },
    )
    with urlopen(req, timeout=20, context=SSL_CONTEXT) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, access_token: str, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        return _get_json(url, access_token, extra_headers=extra_headers)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google API request failed ({exc.code}): {details}") from exc


def post_json(url: str, access_token: str, data: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(data).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "AgentHub-Studio",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20, context=SSL_CONTEXT) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google API request failed ({exc.code}): {details}") from exc


def _new_state(provider: str) -> str:
    state = secrets.token_urlsafe(32)
    _write_json(_state_path(provider), {"state": state, "created_at": time.time()})
    return state


def _validate_state(provider: str, incoming_state: str) -> None:
    stored = _read_json(_state_path(provider))
    if not stored or stored.get("state") != incoming_state:
        raise ValueError("OAuth state did not match. Please try connecting again.")
    if time.time() - float(stored.get("created_at", 0)) > 900:
        raise ValueError("OAuth state expired. Please try connecting again.")


def google_configured() -> bool:
    return bool(config.GOOGLE_CLIENT_ID and config.GOOGLE_CLIENT_SECRET)


def google_login_url() -> str:
    if not google_configured():
        raise OAuthConfigError("Google OAuth is missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET.")
    params = {
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": config.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(config.GOOGLE_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": _new_state("google"),
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def finish_google_callback(code: str, state: str) -> None:
    _validate_state("google", state)
    token = _post_form(
        GOOGLE_TOKEN_URL,
        {
            "code": code,
            "client_id": config.GOOGLE_CLIENT_ID,
            "client_secret": config.GOOGLE_CLIENT_SECRET,
            "redirect_uri": config.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
    )
    token["provider"] = "google"
    token["saved_at"] = time.time()
    token["profile"] = _safe_google_profile(token.get("access_token", ""))
    _write_json(_token_path("google"), token)


def google_profile() -> Dict[str, Any]:
    token = _read_json(_token_path("google")) or {}
    return token.get("profile", {})


def google_access_token() -> str:
    token = _read_json(_token_path("google"))
    if not token:
        raise RuntimeError("Google is not connected.")

    expires_in = int(token.get("expires_in", 0) or 0)
    saved_at = float(token.get("saved_at", 0) or 0)
    if token.get("access_token") and (not expires_in or time.time() < saved_at + expires_in - 60):
        return token["access_token"]

    refresh_token = token.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("Google token expired and no refresh token is available. Disconnect and reconnect Google.")

    refreshed = _post_form(
        GOOGLE_TOKEN_URL,
        {
            "client_id": config.GOOGLE_CLIENT_ID,
            "client_secret": config.GOOGLE_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )
    token.update(refreshed)
    token["refresh_token"] = refresh_token
    token["saved_at"] = time.time()
    token["profile"] = token.get("profile") or _safe_google_profile(token.get("access_token", ""))
    _write_json(_token_path("google"), token)
    return token["access_token"]


def _safe_google_profile(access_token: str) -> Dict[str, Any]:
    if not access_token:
        return {}
    try:
        profile = _get_json(GOOGLE_USERINFO_URL, access_token)
        return {
            "email": profile.get("email"),
            "name": profile.get("name"),
            "picture": profile.get("picture"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch Google profile: %s", exc)
        return {}


def disconnect(provider: str) -> None:
    if provider != "google":
        raise ValueError("Unknown provider.")
    _token_path(provider).unlink(missing_ok=True)
    _state_path(provider).unlink(missing_ok=True)


def is_connected(provider: str) -> bool:
    return _read_json(_token_path(provider)) is not None


def is_tool_connected(tool_key: str) -> bool:
    if tool_key in GOOGLE_TOOL_KEYS:
        return is_connected("google")
    return False


def integration_status() -> Dict[str, Any]:
    google_token = _read_json(_token_path("google"))
    return {
        "google": {
            "configured": google_configured(),
            "connected": google_token is not None,
            "account": (google_token or {}).get("profile", {}),
            "tools": ["Gmail", "Google Drive", "Google Calendar"],
        }
    }
