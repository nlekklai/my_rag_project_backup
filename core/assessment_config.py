import os

# --- Global Paths ---
ASSESSMENT_DIR = "assessment_data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSESSMENT_DIR, exist_ok=True)

# --- FOLDER DEFINITIONS ---
# โฟลเดอร์สำหรับเอกสาร 4 ประเภทหลัก
FOLDERS = {
    "rubrics": os.path.join(ASSESSMENT_DIR, "rubrics"),
    "qa": os.path.join(ASSESSMENT_DIR, "qa"),
    "feedback": os.path.join(ASSESSMENT_DIR, "feedback"),
    "evidence": os.path.join(ASSESSMENT_DIR, "evidence"),
}

# ตรวจสอบและสร้างโฟลเดอร์ย่อยทั้งหมด
for folder_name in FOLDERS:
    os.makedirs(FOLDERS[folder_name], exist_ok=True)
