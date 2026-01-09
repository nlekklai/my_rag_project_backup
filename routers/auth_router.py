# -*- coding: utf-8 -*-
import logging
import platform
import os
from typing import Dict, List, Optional
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status, Form
from pydantic import BaseModel, EmailStr, Field

# 🎯 เชื่อมต่อกับ Config ส่วนกลาง
from config.global_vars import DEFAULT_YEAR, SUPPORTED_ENABLERS

logger = logging.getLogger(__name__)

# ------------------- Pydantic Models -------------------

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    tenant: str = Field(..., description="รหัสองค์กร เช่น pea, tcg")
    enablers: List[str] = Field(default_factory=list, description="รายการ Enabler ที่เข้าถึงได้")
    # 🎯 ดึงปีเริ่มต้นจาก global_vars
    year: str = Field(default=str(DEFAULT_YEAR), description="ปีงบประมาณที่ใช้งาน")
    
class UserRegister(UserBase):
    password: str = Field(..., min_length=8)

class UserMe(UserBase):
    id: str
    is_active: bool = True

class UserDB(UserMe):
    password: str

# ------------------- In-memory DB (simulation) -------------------
USERS: Dict[str, UserDB] = {
    "dev.admin@pea.com": UserDB(
        id="pea-admin-id",
        email="dev.admin@pea.com",
        full_name="Dev Admin (PEA)",
        tenant="pea",
        is_active=True,
        password="P@ssword2568",
        enablers=["KM", "IM", "SP", "SCM", "CG"],
        year=str(DEFAULT_YEAR)
    ),
    "admin@tcg.or.th": UserDB(
        id="tcg-admin-id",
        email="admin@tcg.or.th",
        full_name="Admin TCG",
        tenant="tcg",
        is_active=True,
        password="P@ssword2568",
        enablers=SUPPORTED_ENABLERS, # ให้สิทธิ์ทุกตัวที่มีในระบบ
        year=str(DEFAULT_YEAR)
    )
}

# ------------------- 🟢 Intelligent Persistent Session -------------------
# แก้ปัญหา Server Restart แล้ว Session หลุด (รองรับทั้ง macOS และ Linux Server)
SESSION_FILE = ".dev_session"

def get_persisted_session() -> Optional[str]:
    """ดึง Session ล่าสุดจากไฟล์เพื่อกู้คืนสถานะ Login"""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                email = f.read().strip()
                return email if email in USERS else None
        except Exception as e:
            logger.error(f"Error reading session file: {e}")
            return None
    return None

def save_persisted_session(email: str):
    """บันทึก Session ลงไฟล์ (Persistent Storage)"""
    try:
        with open(SESSION_FILE, "w") as f:
            f.write(email)
    except Exception as e:
        logger.error(f"Save session failed: {e}")

# โหลดสถานะล่าสุดทันทีที่ Start Server
CURRENT_SESSION_USER: Optional[str] = get_persisted_session()

# ------------------- Utility/Mock Dependencies -------------------

async def get_current_user() -> UserMe:
    """Dependency สำหรับดึงข้อมูล User ปัจจุบัน (Auto-Restore จากไฟล์)"""
    global CURRENT_SESSION_USER
    
    # 1. ถ้า RAM ว่าง ให้ลองกู้จากไฟล์ (กัน Error 401 หลัง Restart)
    if not CURRENT_SESSION_USER:
        CURRENT_SESSION_USER = get_persisted_session()
        if CURRENT_SESSION_USER:
            logger.info(f"🔄 [Auth] Session restored for: {CURRENT_SESSION_USER}")

    # 2. ถ้ายังไม่มีอีก แสดงว่ายังไม่ได้ Login จริงๆ
    if not CURRENT_SESSION_USER:
        logger.warning("🚫 [Auth] Access denied: No active session found.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="เซสชันหมดอายุ กรุณา Login ใหม่อีกครั้ง",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 3. ตรวจสอบใน DB
    user_db = USERS.get(CURRENT_SESSION_USER)
    if not user_db or not user_db.is_active:
        raise HTTPException(status_code=403, detail="บัญชีผู้ใช้ไม่มีสิทธิ์เข้าถึง")
            
    return UserMe(**user_db.model_dump(exclude={"password"}))


def check_user_permission(user: UserMe, tenant: str, enabler: Optional[str] = None) -> bool:
    """ตรวจสอบสิทธิ์ (Authorization Gatekeeper)"""
    try:
        target_tenant = str(tenant).strip().lower()
        user_tenant = str(user.tenant).strip().lower()
        
        # 🛡️ 1. ตรวจสอบ Tenant
        if target_tenant != user_tenant:
            logger.error(f"🚫 [Permission Denied] User Tenant:{user_tenant} != Target:{target_tenant}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access Denied: คุณไม่มีสิทธิ์เข้าถึงข้อมูลขององค์กร {tenant}"
            )

        # 🛡️ 2. ตรวจสอบ Enabler (ถ้าส่งมาเช็ค)
        if enabler:
            target_en = str(enabler).strip().upper()
            user_enablers = [str(e).strip().upper() for e in user.enablers]
            
            if target_en not in user_enablers:
                logger.error(f"🚫 [Permission Denied] Enabler mismatch! User has:{user_enablers}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access Denied: คุณไม่มีสิทธิ์ใช้งานในระบบ {target_en}"
                )

        return True

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"💥 [Permission Error]: {str(e)}")
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในการตรวจสอบสิทธิ์")


# ------------------- Router Setup -------------------
auth_router = APIRouter(prefix="/api/auth", tags=["Auth"])

# ------------------- Endpoints -------------------

@auth_router.post("/jwt/login")
async def login_for_access_token(
    username: str = Form(...),
    password: str = Form(...),
):
    global CURRENT_SESSION_USER
    
    input_user = username.strip().lower()
    input_pass = password.strip()
    
    user = USERS.get(input_user)
    
    if not user or user.password != input_pass:
        raise HTTPException(status_code=401, detail="อีเมลหรือรหัสผ่านไม่ถูกต้อง")
    
    # บันทึกสถานะทั้งใน RAM และ File
    CURRENT_SESSION_USER = input_user
    save_persisted_session(input_user)
    
    logger.info(f"✅ Success: Logged in as '{input_user}' (Tenant: {user.tenant})")
    
    return {
        "access_token": f"token_{user.id}",
        "token_type": "bearer",
        "user": user.model_dump(exclude={"password"})
    }

@auth_router.get("/me", response_model=UserMe)
async def read_users_me(current_user: UserMe = Depends(get_current_user)):
    return current_user

@auth_router.post("/logout")
async def logout():
    global CURRENT_SESSION_USER
    CURRENT_SESSION_USER = None
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    logger.info("🚪 User logged out.")
    return {"status": "success", "message": "Logged out"}

@auth_router.post("/register", response_model=UserMe)
async def register_user(user_data: UserRegister):
    if user_data.email in USERS:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = UserDB(id=uuid4().hex, **user_data.model_dump())
    USERS[new_user.email] = new_user
    return UserMe(**new_user.model_dump(exclude={"password"}))