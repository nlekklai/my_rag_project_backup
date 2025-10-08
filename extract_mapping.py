import re
import os
import fitz
import json
from collections import defaultdict

# --- โครงสร้าง KM ---
KM_STRUCTURE = {
    "1. การนำองค์กร": ["1.1 วิสัยทัศน์", "1.2 การมีส่วนร่วมของผู้บริหาร"],
    "2. การวางแผนและทรัพยากรสนับสนุน": ["2.1 การวางแผนและทรัพยากรสนับสนุน", "2.2 การจัดสรรทรัพยากร"],
    "3. บุคลากร": ["3.1 ความตระหนักความเข้าใจการมีส่วนร่วม และการสร้างแรงจูงใจด้าน KM",
                    "3.2 วัฒนธรรมและสภาพแวดล้อมการทำงาน",
                    "3.3 ความสามารถและความรับผิดชอบของทีมงาน"],
    "4. กระบวนการจัดการความรู้": ["4.1 กระบวนการจัดการความรู้ที่เป็นระบบและมีการนำเทคโนโลยีมาประยุกต์ใช้",
                                   "4.2 สารสนเทศ/ความรู้จากหน่วยงานภายนอก"],
    "5. กระบวนการปฏิบัติ": ["5.1 การปฏิบัติงานโดยใช้ความรู้เป็นฐาน",
                             "5.2 การสร้างความตระหนักเรื่องความเสี่ยงโดยการใช้ความรู้เป็นฐาน"],
    "6. ผลลัพธ์ของการจัดการความรู้": ["6.1 ผลงานด้านต่างๆที่เกี่ยวข้องกับการจัดการความรู้"]
}

ALL_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]

def normalize_filename(name: str) -> str:
    """Clean filenames — remove spaces in UUID, merge broken tokens"""
    name = re.sub(r"\s+", "", name) if re.match(r"[0-9a-fA-F\-]{10,}", name) else re.sub(r"\s+", " ", name)
    return name.strip()

def normalize_question(text: str) -> str:
    """Normalize question labels"""
    if "ระดับ" in text:
        return text.replace("ระดับ", "Level ").strip()
    elif "ข้อ" in text:
        return text.replace("ข้อ", "Clause ").strip()
    return text.strip()

def extract_semantic_mapping_by_category(pdf_path, organization="PEA", section="KM"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    results = defaultdict(lambda: {"evidence": set(), "pages": set(), "category": None, "subtopic": None})
    current_category = None
    current_subtopic = None
    current_level = None
    pending_evidence = []

    for page in doc:
        # Merge text, normalize spaces
        text = re.sub(r"\s+", " ", page.get_text("text"))

        # ตรวจสอบหมวดใหญ่
        for cat in KM_STRUCTURE.keys():
            if cat in text:
                current_category = cat
                break

        # ตรวจสอบหมวดย่อย
        if current_category:
            for sub in KM_STRUCTURE[current_category]:
                if sub in text:
                    current_subtopic = sub
                    break

        # ตรวจสอบไฟล์ evidence (robust)
        files = re.findall(r"([^\s]+?\.(?:pdf|xlsx|jpg|png|docx))", text, flags=re.IGNORECASE)
        clean_files = [normalize_filename(f) for f in files if len(f) < 120]
        pending_evidence.extend(clean_files)

        # ตรวจสอบ Level/ข้อ
        level_match = re.findall(r"(ระดับ\s*\d+|ข้อ\s*\d+\.?\d*)", text)
        if level_match:
            current_level = level_match[-1].strip()
            # assign pending evidence
            if current_level and pending_evidence:
                key = f"{current_category}__{current_subtopic}__{current_level}"
                results[key]["evidence"].update(pending_evidence)
                results[key]["pages"].add(page.number + 1)
                results[key]["category"] = current_category
                results[key]["subtopic"] = current_subtopic
                pending_evidence = []

    # สร้าง JSON-ready list ครบทุก Level/Subtopic
    final = []
    for cat, subtopics in KM_STRUCTURE.items():
        for sub in subtopics:
            for lvl in ALL_LEVELS:
                key = f"{cat}__{sub}__ระดับ {lvl.split()[1]}"  # ระดับ X
                data = results.get(key, {"evidence": set(), "pages": set(), "category": cat, "subtopic": sub})
                final.append({
                    "id": f"{organization}_{cat.replace(' ', '')}_{sub.replace(' ', '')}_{lvl.replace(' ', '')}",
                    "organization": organization,
                    "section": section,
                    "category": cat,
                    "subtopic": sub,
                    "level": lvl,
                    "evidence_files": sorted(data["evidence"]),
                    "page_numbers": sorted(data["pages"]),
                    "evidence_count": len(data["evidence"])
                })

    return final

if __name__ == "__main__":
    pdf_path = os.path.join("data", "evidence", "PEA คำอธิบาย 2567.pdf")
    data = extract_semantic_mapping_by_category(pdf_path)
    os.makedirs("output", exist_ok=True)
    with open("output/mappings_pea_by_category.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Extracted {len(data)} mappings → output/mappings_pea_by_category.json")
