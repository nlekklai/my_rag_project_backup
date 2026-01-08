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
    enablers: List[str] = Field(default_factory=list, description="รายการ Enabler ที่ User นี้เข้าถึงได้")
    
class UserRegister(UserBase):
    password: str = Field(..., min_length=8)

class UserMe(UserBase):
    id: str
    is_active: bool = True

class UserDB(UserMe):
    password: str

# ------------------- In-memory DB (simulation) -------------------
# ประกาศและใส่ข้อมูลลงไปทันทีในขั้นตอนเดียว
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
# 🟢 ตัวแปรจำลอง Session (เพราะรัน Local และยังไม่มีระบบ Token จริงที่ซับซ้อน)
CURRENT_SESSION_USER: Optional[str] = None

# ------------------- Utility/Mock Dependencies -------------------

async def get_current_user() -> UserMe:
    """
    ดึงข้อมูล User ปัจจุบันจาก Session ในหน่วยความจำ (Global Variable)
    และตรวจสอบสิทธิ์ก่อนอนุญาตให้เข้าถึง API
    """
    global CURRENT_SESSION_USER
    
    # 🎯 1. ตรวจสอบว่ามี User Login อยู่ในระบบจริงหรือไม่
    # เราลบ "or admin@tcg.or.th" ออก เพื่อป้องกันการสวมรอย Identity อัตโนมัติ
    if not CURRENT_SESSION_USER:
        logger.warning("🚫 [Auth] Access denied: No active session found.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="เซสชันหมดอายุหรือยังไม่ได้เข้าสู่ระบบ กรุณา Login ใหม่อีกครั้ง",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email = CURRENT_SESSION_USER
    
    # 🎯 2. ตรวจสอบว่า Email ใน Session มีอยู่ใน Database หรือไม่
    if email in USERS:
        user_db = USERS[email]
        
        # ตรวจสอบว่าบัญชีถูกระงับหรือไม่
        if not user_db.is_active:
            logger.error(f"❌ [Auth] User account is inactive: {email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="บัญชีผู้ใช้นี้ถูกระงับการใช้งาน"
            )
            
        # คืนค่าเป็น UserMe Object (ตัด Password ออก)
        logger.info(f"✅ [Auth] Identity Verified: {email} (Tenant: {user_db.tenant})")
        return UserMe(**user_db.model_dump(exclude={"password"}))

    # 🎯 3. กรณีมี Session แต่หา User ไม่เจอใน DB (เช่น มีการแก้โค้ด USERS)
    logger.error(f"❌ [Auth] Session exists for {email} but user not found in DB.")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="ไม่พบข้อมูลผู้ใช้ในระบบ",
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
    return UserMe(**new_user.model_dump(exclude={"password"}))

@auth_router.post("/jwt/login")
async def login_for_access_token(
    username: str = Form(...),
    password: str = Form(...),
):
    global CURRENT_SESSION_USER
    
    # 1. ล้างขยะออกจาก input
    input_user = username.strip().lower()
    input_pass = password.strip()
    
    # 2. ดูค่าใน USERS จริงๆ ณ วินาทีที่กด Login
    # (ถ้าพิมพ์ออกมาแล้วเป็น {} แสดงว่า Dictionary ว่างเปล่า)
    print(f"\n--- DEBUG LOGIN ---")
    print(f"Current DB Keys: {list(USERS.keys())}")
    print(f"Searching for: '{input_user}'")
    
    # 3. ลองดึงข้อมูล
    user = USERS.get(input_user)
    
    if not user:
        # ลองค้นหาแบบ Manual (เผื่อมีช่องว่างหลุดใน Dictionary)
        for key in USERS.keys():
            if key.strip().lower() == input_user:
                user = USERS[key]
                break
    
    if not user:
        print(f"❌ Error: '{input_user}' not found in DB")
        raise HTTPException(status_code=401, detail="Incorrect username or password")
        
    if user.password != input_pass:
        print(f"❌ Error: Password mismatch for '{input_user}'")
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    print(f"✅ Success: Logged in as '{input_user}'")
    CURRENT_SESSION_USER = input_user
    
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
    return {"status": "success", "message": "Logged out"}