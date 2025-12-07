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
    year: int = Field(..., example=2568, description="ปีงบประมาณ")
    # 🟢 FIX: เพิ่ม Field สำหรับรายการ Enabler ที่ User เข้าถึงได้
    enablers: List[str] = Field(default_factory=list, description="รายการ Enabler ที่ User นี้เข้าถึงได้")
    
class UserRegister(UserBase):
    password: str = Field(..., min_length=8)

class UserMe(UserBase):
    id: str
    is_active: bool = True

class UserDB(UserMe):
    password: str

# ------------------- In-memory DB (for simulation) -------------------
# NOTE: ใน Production ควรใช้ Database
USERS: Dict[str, UserDB] = {}

# Seed initial user for testing (ตามข้อมูลใน UI)
USERS["dev.admin@pea.com"] = UserDB(
    id="dev-admin-id",
    email="dev.admin@pea.com",
    full_name="Dev Admin (PEA)",
    tenant="pea",
    year=2568,
    is_active=True,
    password="P@ssword2568",
    # 🟢 FIX: กำหนดสิทธิ์ Enabler ให้กับ User นี้
    enablers=["KM","IM"] 
)

# ------------------- Utility/Mock Dependencies -------------------

# Mock function for token creation (ไม่ต้องใช้งานจริงในตัวอย่างนี้)
def create_access_token(data: dict, expires_delta: float):
    return "MOCK_JWT_TOKEN" 

# Dependency to get the current user (จำลองการตรวจสอบสิทธิ์)
async def get_current_user() -> UserMe:
    # ⚠️ คำเตือน: ใน Production จะต้องใช้ OAuth2PasswordBearer และตรวจสอบ JWT Token จริง
    
    # สำหรับการจำลองใน Dev Environment จะคืนค่า Test User เสมอเมื่อมีการเรียก
    if "dev.admin@pea.com" in USERS:
        user = USERS["dev.admin@pea.com"]
        # UserMe จะมี Field 'enablers' อยู่แล้ว
        return UserMe(**user.model_dump(exclude={"password"}))

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials (Mocked: No test user found)",
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
        year=user_data.year, 
        is_active=True,
        # Field enablers จะถูกตั้งค่าตามค่า default หรือค่าที่ส่งมาใน Form
        enablers=user_data.enablers, 
        password=user_data.password
    )
    
    USERS[new_user.email] = new_user
    logger.info(f"New user registered: {new_user.email} for {new_user.tenant}/{new_user.year}")
    
    # ส่งข้อมูล User ที่ไม่มี Password กลับไป (รวมถึง Field enablers)
    return UserMe(**new_user.model_dump(exclude={"password"}))

# ------------------- Login Endpoint (FINAL FIX for Frontend) -------------------
@auth_router.post("/jwt/login")
async def login_for_access_token(
    # 🌟 แก้ไขให้รับ 'username' เพื่อให้ตรงกับ formData.append('username', email) ใน Frontend 🌟
    username: str = Form(..., example="dev.admin@pea.com"),
    password: str = Form(..., example="P@ssword2568"),
):
    # ใช้ username ที่รับมาในการค้นหา User
    user = USERS.get(username)
    
    if not user or user.password != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # สร้าง UserMe object ซึ่งจะมี Field 'enablers' ติดมาด้วย
    user_data_me = UserMe(**user.model_dump(exclude={"password"}))

    # Mock Token generation
    access_token = f"simulated_jwt_token_for_{user.id}"
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        # 🟢 FIX: ส่ง User Context กลับไปด้วย ซึ่งมี Field enablers
        "user": user_data_me.model_dump() 
    }

@auth_router.get("/me", response_model=UserMe)
async def read_users_me(current_user: UserMe = Depends(get_current_user)):
    # คืนค่าข้อมูล User พร้อม Tenant/Year/Enablers Context
    return current_user