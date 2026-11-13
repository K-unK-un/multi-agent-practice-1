"""管理员接口：用户管理（列表 / 设等级 / 禁用启用 / 重置密码）。仅 role=admin 可用。

权限模型：
- 管理员：最高权限，可查看与修改任何内容、给普通用户设置 1~4 级；
- 4 级普通用户：查看权限与管理员相同（未来 RAG 分级后可见全部文档），但无任何修改权限——
  本文件所有接口均以 require_admin（role 校验）把关，等级高不等于管理权。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.core.logger import get_logger
from app.core.security import hash_password, require_admin
from app.store.mysql import get_mysql

logger = get_logger("api.admin")
router = APIRouter(prefix="/admin", tags=["admin"])

_LEVEL_MIN, _LEVEL_MAX = 1, 4


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    """对外暴露的用户信息（不含密码哈希）。"""
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


class UserUpdateRequest(BaseModel):
    """可选字段更新：设等级 / 禁用启用 / 重置密码。"""

    level: int | None = Field(default=None, ge=_LEVEL_MIN, le=_LEVEL_MAX)
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=6, max_length=64)

    @field_validator("level")
    @classmethod
    def _check_level(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if not _LEVEL_MIN <= value <= _LEVEL_MAX:
            raise ValueError(f"等级需在 {_LEVEL_MIN}~{_LEVEL_MAX} 之间")
        return value


@router.get("/users")
def list_users(admin: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    """全部用户列表（含管理员自身）。"""
    try:
        users = get_mysql().list_users()
    except Exception:
        logger.exception("管理员查询用户列表失败")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="用户存储不可用")
    return {"users": [_public_user(u) for u in users], "total": len(users)}


@router.patch("/users/{user_id}")
def update_user(
    user_id: int,
    body: UserUpdateRequest,
    admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """设置用户等级 / 禁用启用 / 重置密码。管理员自身仅允许改密码（等级恒为 4）。"""
    if not any((body.level is not None, body.is_active is not None, body.password)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="没有需要更新的字段")

    store = get_mysql()
    try:
        target = store.get_user_by_id(user_id)
    except Exception:
        logger.exception("管理员查询目标用户失败 user_id=%s", user_id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="用户存储不可用")

    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")

    is_admin_target = (target.get("role") == "admin")
    if is_admin_target and body.level is not None and body.level != 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="管理员等级恒为 4，不可修改"
        )

    try:
        updated = store.update_user(
            user_id,
            level=body.level,
            is_active=body.is_active,
            password_hash=hash_password(body.password) if body.password else None,
        )
    except Exception:
        logger.exception("管理员更新用户失败 user_id=%s", user_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="更新失败")

    logger.info(
        "管理员 %s 更新用户 user_id=%s level=%s is_active=%s 改密=%s",
        admin.get("username"), user_id, body.level, body.is_active, bool(body.password),
    )
    return {"ok": True, "user": _public_user(updated)}
