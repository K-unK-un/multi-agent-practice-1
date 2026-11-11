"""认证与密码安全：bcrypt 密码哈希、PyJWT 令牌签发/校验、FastAPI 鉴权依赖。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("core.security")

# bcrypt 5.0 起传入超过 72 字节的密码会直接抛 ValueError，这里统一按字节截断。
_BCRYPT_MAX_BYTES = 72

# auto_error=False：是否放行由 require_auth 决定（auth_enabled 关闭时允许匿名）。
_bearer_scheme = HTTPBearer(auto_error=False)

ANONYMOUS_USER: dict[str, Any] = {"username": "anonymous", "scopes": ["employee"]}


def _password_bytes(password: str) -> bytes:
    """把明文密码转为 bcrypt 可接受的 bytes，并限制在 72 字节内。"""
    raw = password.encode("utf-8")
    if len(raw) > _BCRYPT_MAX_BYTES:
        raw = raw[:_BCRYPT_MAX_BYTES]
    return raw


def hash_password(pw: str) -> str:
    """生成 bcrypt 哈希（含随机盐），返回 UTF-8 字符串。"""
    return bcrypt.hashpw(_password_bytes(pw), bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    """校验明文密码与 bcrypt 哈希；哈希损坏或格式异常时返回 False。"""
    if not pw or not hashed:
        return False
    try:
        return bcrypt.checkpw(_password_bytes(pw), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        logger.warning("密码哈希校验失败：哈希格式非法")
        return False


def create_access_token(username: str, scopes: list[str]) -> str:
    """签发 JWT：sub/username、scopes、exp = now + jwt_expire_minutes。"""
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": username,
        "username": username,
        "scopes": list(scopes or []),
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any] | None:
    """解码并校验 JWT；过期、签名错误等任何无效情况均返回 None。"""
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        return None
    return payload if isinstance(payload, dict) else None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, Any]:
    """FastAPI 依赖：校验 Authorization: Bearer <token>。

    - auth_enabled=False：直接返回匿名用户（开发模式）；
    - auth_enabled=True：缺少/无效/过期 token 一律 401。
    """
    if not settings.auth_enabled:
        return dict(ANONYMOUS_USER)

    if credentials is None or (credentials.scheme or "").lower() != "bearer":
        raise _unauthorized("缺少 Bearer 认证令牌")

    payload = decode_token(credentials.credentials)
    if payload is None:
        raise _unauthorized("认证令牌无效或已过期")

    username = payload.get("username") or payload.get("sub")
    if not username:
        raise _unauthorized("认证令牌缺少用户标识")

    scopes = payload.get("scopes")
    return {
        "username": str(username),
        "scopes": [str(s) for s in scopes] if isinstance(scopes, (list, tuple)) else [],
    }
