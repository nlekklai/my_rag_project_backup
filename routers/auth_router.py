# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status, Form
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

# ------------------- Pydantic Models -------------------

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    tenant: str = Field(..., example="pea", description="รหัสองค์กร")
    # 🔴 REMOVED: year (เราจะไม่ผูก user กับ year อีกต่อไป)
    enablers: List[str] = Field(default_factory=list, description="รายการ Enabler ที่ User นี้เข้าถึงได้")
    
class UserRegister(UserBase):
    password: str = Field(..., min_length=8)

class UserMe(UserBase):
    id: str
    is_active: bool = True

class UserDB(UserMe):
    password: str

# ------------------- In-memory DB (simulation) -------------------
USERS: Dict[str, UserDB] = {}

# Seed initial user for testing
USERS["dev.admin@pea.com"] = UserDB(
    id="dev-admin-id",
    email="dev.admin@pea.com",
    full_name="Dev Admin (PEA)",
    tenant="pea",
    # 🔴 REMOVED: year=2568
    is_active=True,
    password="P@ssword2568",
    enablers=["KM","IM"] 
)

# ------------------- Utility/Mock Dependencies -------------------

async def get_current_user() -> UserMe:
    # สำหรับการจำลองใน Dev Environment จะคืนค่า Test User เสมอเมื่อมีการเรียก
    if "dev.admin@pea.com" in USERS:
        user = USERS["dev.admin@pea.com"]
        return UserMe(**user.model_dump(exclude={"password"}))

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

# ------------------- Router Setup -------------------
auth_router = APIRouter(prefix="/api/auth", tags=["Auth"])

# ------------------- Endpoints -------------------

@auth_router.post("/register", response_model=UserMe, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister):
    if user_data.email in USERS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    new_user_id = uuid4().hex
    
    # 🟢 สร้าง User ใหม่โดยไม่มี Field 'year'
    new_user = UserDB(
        id=new_user_id,
        email=user_data.email,
        full_name=user_data.full_name,
        tenant=user_data.tenant, 
        is_active=True,
        enablers=user_data.enablers, 
        password=user_data.password
    )
    
    USERS[new_user.email] = new_user
    # Log เฉพาะ Tenant เพราะปีจะถูกเลือกจากหน้าจอเป็นครั้งๆ ไป
    logger.info(f"New user registered: {new_user.email} for tenant: {new_user.tenant}")
    
    return UserMe(**new_user.model_dump(exclude={"password"}))

@auth_router.post("/jwt/login")
async def login_for_access_token(
    username: str = Form(..., example="dev.admin@pea.com"),
    password: str = Form(..., example="P@ssword2568"),
):
    user = USERS.get(username)
    
    if not user or user.password != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_data_me = UserMe(**user.model_dump(exclude={"password"}))
    access_token = f"simulated_jwt_token_for_{user.id}"
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_data_me.model_dump() 
    }

@auth_router.get("/me", response_model=UserMe)
async def read_users_me(current_user: UserMe = Depends(get_current_user)):
    # คืนค่าข้อมูล User โดยใน object นี้จะไม่มี field 'year' แล้ว
    return current_user