"""认证接口：登录签发 JWT；开发模式（auth_enabled=False）开放注册。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.core.logger import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.store.mysql import get_mysql

logger = get_logger("api.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """兼容存储层返回 dict 或 ORM 对象两种情况。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)

    @field_validator("username")
    @classmethod
    def _strip_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("用户名不能为空")
        return value


class RegisterRequest(LoginRequest):
    scopes: list[str] = Field(default_factory=lambda: ["employee"])

    @field_validator("scopes")
    @classmethod
    def _clean_scopes(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if item and str(item).strip()]
        return list(dict.fromkeys(cleaned)) or ["employee"]


@router.post("/login")
def login(body: LoginRequest) -> dict[str, Any]:
    """用户名 + 密码登录，成功返回 JWT 与 scope 列表。"""
    try:
        user = get_mysql().get_user(body.username)
    except Exception:
        logger.exception("登录时查询用户失败 username=%s", body.username)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="用户存储不可用")

    password_hash = _field(user, "password_hash") if user else None
    if not user or not password_hash or not verify_password(body.password, str(password_hash)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scopes = [str(item) for item in (_field(user, "scopes") or []) if item]
    if not scopes:
        scopes = ["employee"]

    token = create_access_token(body.username, scopes)
    return {
        "token": token,
        "token_type": "bearer",
        "scopes": scopes,
        "username": body.username,
    }


@router.post("/register")
def register(body: RegisterRequest) -> dict[str, Any]:
    """开发模式开放注册；生产模式（auth_enabled=True）返回 403。"""
    if settings.auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="注册仅在开发模式（AUTH_ENABLED=false）开放",
        )

    store = get_mysql()
    try:
        existing = store.get_user(body.username)
    except Exception:
        logger.exception("注册时查询用户失败 username=%s", body.username)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="用户存储不可用")

    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")

    try:
        created = store.create_user(
            username=body.username,
            password_hash=hash_password(body.password),
            scopes=body.scopes,
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(word in message for word in ("duplicate", "unique", "重复")):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")
        logger.exception("创建用户失败 username=%s", body.username)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="注册失败")

    if created is False:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")

    logger.info("注册新用户 username=%s scopes=%s", body.username, body.scopes)
    return {"ok": True, "username": body.username, "scopes": body.scopes}
