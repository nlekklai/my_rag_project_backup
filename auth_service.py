# auth_service.py
from database import SessionLocal, UserTable
from typing import Optional
import json

def get_user_by_username(username: str) -> Optional[UserTable]:
    db = SessionLocal()
    user = db.query(UserTable).filter(UserTable.username == username.lower()).first()
    db.close()
    return user

def create_initial_admin():
    db = SessionLocal()
    
    # รายการ User เริ่มต้นที่ต้องการสร้าง
    initial_users = [
        {
            "id": "pea-admin-id",
            "username": "admin@pea.com",
            "password_hash": "P@ssword2568",
            "full_name": "Dev Admin (PEA)",
            "tenant": "pea",
            "role": "admin",
            "enablers": ["IM", "KM", "CG", "DT"] # สิทธิ์ของ PEA
        },
        {
            "id": "tcg-admin-id",
            "username": "admin@tcg.or.th",
            "password_hash": "P@ssword2568",
            "full_name": "Admin TCG",
            "tenant": "tcg",
            "role": "admin",
            "enablers": ["IM", "KM", "CG", "DT", "HCM"] # สิทธิ์ของ TCG
        }
    ]

    for user_data in initial_users:
        # เช็คก่อนว่า username นี้มีใน DB หรือยัง
        exists = db.query(UserTable).filter(UserTable.username == user_data["username"]).first()
        
        if not exists:
            new_user = UserTable(
                id=user_data["id"],
                username=user_data["username"],
                password_hash=user_data["password_hash"],
                full_name=user_data["full_name"],
                tenant=user_data["tenant"],
                role=user_data["role"],
                # ⚠️ สำคัญ: SQLite ไม่เก็บ List ต้องแปลงเป็น JSON String ก่อน
                enablers=json.dumps(user_data["enablers"]) 
            )
            db.add(new_user)
            print(f"✅ Created initial user: {user_data['username']} for {user_data['tenant']}")
    
    db.commit()
    db.close()

# ใช้ตอน Register (ตัวอย่าง)
def create_new_user(user_id, username, password, full_name, tenant, enablers):
    db = SessionLocal()
    new_user = UserTable(
        id=user_id,
        username=username.lower(),
        password_hash=password, # ควร hash ก่อนเก็บ
        full_name=full_name,
        tenant=tenant,
        enablers=json.dumps(enablers)
    )
    db.add(new_user)
    db.commit()
    db.close()