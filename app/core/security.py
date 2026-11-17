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

ANONYMOUS_USER: dict[str, Any] = {
    "id": None,
    "username": "anonymous",
    "role": "user",
    "level": 1,
    "is_active": True,
    "scopes": ["employee"],
}


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


def create_access_token(user: dict[str, Any]) -> str:
    """签发 JWT：sub=username、uid/role/level、exp = now + jwt_expire_minutes。"""
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user.get("username") or ""),
        "username": str(user.get("username") or ""),
        "uid": int(user.get("id") or 0),
        "role": str(user.get("role") or "user"),
        "level": int(user.get("level") or 1),
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


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, Any]:
    """FastAPI 依赖：校验 Authorization: Bearer <token>，并**实时从数据库**加载用户。

    - auth_enabled=False：返回匿名用户（开发模式）；
    - auth_enabled=True：缺少/无效/过期 token 一律 401；用户被禁用 401；
      管理员修改等级后**无需重新登录**立即生效（token 只证明身份，权限以库为准）。
    """
    if not settings.auth_enabled:
        return dict(ANONYMOUS_USER)

    if credentials is None or (credentials.scheme or "").lower() != "bearer":
        raise _unauthorized("缺少 Bearer 认证令牌")

    payload = decode_token(credentials.credentials)
    if payload is None:
        raise _unauthorized("认证令牌无效或已过期")

    username = payload.get("username") or payload.get("sub")
    uid = payload.get("uid")
    if not username and not uid:
        raise _unauthorized("认证令牌缺少用户标识")

    # 实时查库：拿到最新 role / level / is_active / scopes
    try:
        from app.store.mysql import get_mysql

        store = get_mysql()
        if uid:
            user = store.get_user_by_id(int(uid))
        else:
            user = store.get_user(str(username))
    except Exception:
        logger.exception("鉴权时查询用户失败 username=%s", username)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="用户存储不可用"
        )

    if user is None:
        raise _unauthorized("用户不存在")
    if not user.get("is_active", True):
        raise _unauthorized("账号已被禁用")

    return {
        "id": user.get("id"),
        "username": user.get("username"),
        "role": user.get("role") or "user",
        "level": int(user.get("level") or 1),
        "is_active": bool(user.get("is_active", True)),
        "scopes": list(user.get("scopes") or []),
    }


async def require_admin(
    current_user: dict[str, Any] = Depends(require_auth),
) -> dict[str, Any]:
    """管理员依赖：先鉴权，再校验 role == 'admin'。"""
    if current_user.get("role") != "admin":
        raise _forbidden("需要管理员权限")
    return current_user


async def require_kb_viewer(
    current_user: dict[str, Any] = Depends(require_auth),
) -> dict[str, Any]:
    """知识库只读依赖：管理员或 4 级用户可查看文档列表。

    4 级用户仅拥有“查看”权限，上传 / 重建等写操作仍由 require_admin 把关。
    """
    is_admin = current_user.get("role") == "admin"
    try:
        level = int(current_user.get("level") or 1)
    except (TypeError, ValueError):
        level = 1
    if not is_admin and level < 4:
        raise _forbidden("需要管理员或四级用户权限")
    return current_user
