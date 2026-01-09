# -*- coding: utf-8 -*-
import logging
import platform
import os
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
    enablers: List[str] = Field(default_factory=list, description="รายการ Enabler ที่ User นี้เข้าถึงได้")
    
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
        id="dev-admin-id",
        email="dev.admin@pea.com",
        full_name="Dev Admin (PEA)",
        tenant="pea",
        is_active=True,
        password="P@ssword2568",
        enablers=["KM","IM"] 
    ),
    "admin@tcg.or.th": UserDB(
        id="tcg-admin-id",
        email="admin@tcg.or.th",
        full_name="Admin TCG",
        tenant="tcg",
        is_active=True,
        password="P@ssword2568",
        enablers=["KM", "IM"]
    )
}

# ------------------- 🟢 Intelligent Session for Local/Server -------------------
SESSION_FILE = ".dev_session"
IS_MACOS = platform.system() == "Darwin"

def get_persisted_session() -> Optional[str]:
    """ดึง Session ล่าสุดจากไฟล์ (เฉพาะตอน Dev บน Mac เพื่อป้องกัน Hot-reload แล้วหลุด)"""
    if IS_MACOS and os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                email = f.read().strip()
                return email if email in USERS else "admin@tcg.or.th"
        except:
            return "admin@tcg.or.th"
    return "admin@tcg.or.th" if IS_MACOS else None

def save_persisted_session(email: str):
    """บันทึก Session ลงไฟล์ (เฉพาะตอน Dev บน Mac)"""
    if IS_MACOS:
        with open(SESSION_FILE, "w") as f:
            f.write(email)

# ค่าเริ่มต้นตอนเริ่มระบบ
CURRENT_SESSION_USER: Optional[str] = get_persisted_session()

if IS_MACOS:
    logger.info(f"🛠️ [Auth System] macOS Detected: Auto-login enabled (Current: {CURRENT_SESSION_USER})")

# ------------------- Utility/Mock Dependencies -------------------

async def get_current_user() -> UserMe:
    """
    ดึงข้อมูล User ปัจจุบันจาก Session ในหน่วยความจำ
    รองรับ Auto-login สำหรับ Local Development (macOS)
    """
    global CURRENT_SESSION_USER
    
    # 🎯 1. กรณีไม่มี Session: ถ้าอยู่บน Mac ให้ดึงค่าล่าสุดกลับมา
    if not CURRENT_SESSION_USER:
        if IS_MACOS:
            CURRENT_SESSION_USER = get_persisted_session()
            logger.info(f"🛠️ [Auth] Restoring session: {CURRENT_SESSION_USER}")
        else:
            logger.warning("🚫 [Auth] Access denied: No active session found.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="เซสชันหมดอายุหรือยังไม่ได้เข้าสู่ระบบ กรุณา Login ใหม่อีกครั้ง",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    email = CURRENT_SESSION_USER
    
    # 🎯 2. ตรวจสอบข้อมูลใน DB
    if email in USERS:
        user_db = USERS[email]
        if not user_db.is_active:
            raise HTTPException(status_code=403, detail="บัญชีผู้ใช้นี้ถูกระงับการใช้งาน")
            
        return UserMe(**user_db.model_dump(exclude={"password"}))

    # 🎯 3. กรณีหา User ไม่เจอ
    raise HTTPException(status_code=401, detail="ไม่พบข้อมูลผู้ใช้ในระบบ")


def check_user_permission(user: UserMe, tenant: str, enabler: Optional[str] = None) -> bool:
    """
    ตรวจสอบสิทธิ์การเข้าถึงข้อมูล (Authorization Gatekeeper)
    ใช้ตรวจสอบว่า User มีสิทธิ์ใน Tenant และ Enabler ที่ระบุหรือไม่
    """
    try:
        # Normalize ค่าเพื่อป้องกันปัญหาตัวพิมพ์เล็ก-ใหญ่
        target_tenant = str(tenant).strip().lower()
        user_tenant = str(user.tenant).strip().lower()
        
        # 1. ตรวจสอบ Tenant
        if target_tenant != user_tenant:
            logger.error(f"🚫 [Permission Denied] User Tenant:{user_tenant} != Target:{target_tenant}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access Denied: คุณไม่มีสิทธิ์เข้าถึงข้อมูลขององค์กร {tenant}"
            )

        # 2. ตรวจสอบ Enabler (ถ้าส่งมาเช็ค)
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
    
    # บันทึก Session
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
    if IS_MACOS and os.path.exists(SESSION_FILE):
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