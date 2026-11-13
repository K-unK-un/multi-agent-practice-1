"""认证接口：注册（唯一用户名，自动 1 级）/ 登录 / 当前用户 / 修改密码。"""
from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.core.logger import get_logger
from app.core.security import (
    create_access_token,
    hash_password,
    require_auth,
    verify_password,
)
from app.store.mysql import get_mysql

logger = get_logger("api.auth")
router = APIRouter(prefix="/auth", tags=["auth"])

# 用户名：3~32 位，字母/数字/下划线/中文（不允许空白与特殊符号，避免日志与 URL 污染）
_USERNAME_RE = re.compile(r"^[\w一-龥]{3,32}$")
_PASSWORD_MIN = 6
_PASSWORD_MAX = 64


def _public_user(user: dict[str, Any] | None) -> dict[str, Any]:
    """对外暴露的用户信息（不含密码哈希）。"""
    if not user:
        return {}
    return {
        "id": user.get("id"),
        "username": user.get("username"),
        "role": user.get("role") or "user",
        "level": int(user.get("level") or 1),
        "is_active": bool(user.get("is_active", True)),
        "scopes": list(user.get("scopes") or []),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
    }


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def _strip_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("用户名不能为空")
        return value


class RegisterRequest(BaseModel):
    """注册：仅用户名 + 密码。新用户自动 role=user、level=1。"""

    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)

    @field_validator("username")
    @classmethod
    def _check_username(cls, value: str) -> str:
        value = value.strip()
        if not _USERNAME_RE.match(value):
            raise ValueError("用户名需为 3~32 位字母、数字、下划线或中文")
        return value


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)


@router.post("/login")
def login(body: LoginRequest) -> dict[str, Any]:
    """用户名 + 密码登录，成功返回 JWT 与用户信息（含 role / level）。"""
    try:
        user = get_mysql().get_user(body.username)
    except Exception:
        logger.exception("登录时查询用户失败 username=%s", body.username)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="用户存储不可用")

    password_hash = user.get("password_hash") if user else None
    if not user or not password_hash or not verify_password(body.password, str(password_hash)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号已被禁用，请联系管理员",
        )

    token = create_access_token(user)
    get_mysql().touch_login(str(user.get("username")))
    return {"token": token, "token_type": "bearer", "user": _public_user(user)}


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest) -> dict[str, Any]:
    """注册普通用户：用户名全局唯一，自动 1 级。成功即返回 token（自动登录）。"""
    store = get_mysql()
    try:
        existing = store.get_user(body.username)
    except Exception:
        logger.exception("注册时查询用户失败 username=%s", body.username)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="用户存储不可用")

    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")

    try:
        user = store.create_user(
            username=body.username,
            password_hash=hash_password(body.password),
            role="user",
            level=1,
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(word in message for word in ("duplicate", "unique", "重复")):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")
        logger.exception("创建用户失败 username=%s", body.username)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="注册失败")

    logger.info("注册新用户 username=%s level=1", body.username)
    return {"token": create_access_token(user), "token_type": "bearer", "user": _public_user(user)}


@router.get("/me")
def me(current_user: dict[str, Any] = Depends(require_auth)) -> dict[str, Any]:
    """当前登录用户信息（实时从库读取，等级修改立即生效）。"""
    return current_user


@router.post("/change-password")
def change_password(
    body: ChangePasswordRequest,
    current_user: dict[str, Any] = Depends(require_auth),
) -> dict[str, Any]:
    """修改自己的密码（需验证旧密码）。"""
    store = get_mysql()
    try:
        user = store.get_user(str(current_user.get("username")))
    except Exception:
        logger.exception("改密时查询用户失败")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="用户存储不可用")

    if not user or not verify_password(body.old_password, str(user.get("password_hash") or "")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="旧密码不正确")

    try:
        store.update_user(int(user["id"]), password_hash=hash_password(body.new_password))
    except Exception:
        logger.exception("修改密码失败 username=%s", current_user.get("username"))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="修改密码失败")

    return {"ok": True, "message": "密码已更新"}
