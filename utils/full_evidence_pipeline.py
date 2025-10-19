import os
import re
import json
import fitz
from typing import List, Dict, Any, Set

# --- Cleaning Helper Functions ---

def is_uuid_like(name: str) -> bool:
    """
    ตรวจสอบว่าชื่อไฟล์มีลักษณะคล้าย UUID (ยาว 20 ตัวขึ้นไป) ที่มีนามสกุล
    """
    return bool(re.fullmatch(r"[0-9a-fA-F\-]{20,}\.(pdf|jpg|png|docx|xlsx|pptx|zip)", name.strip()))

def clean_filename(name: str) -> str:
    """
    ทำความสะอาดชื่อไฟล์: ยุบช่องว่าง และลบช่องว่างทั้งหมดหากดูเหมือน UUID
    """
    # If it contains many hex characters, treat it as a UUID and remove spaces
    if re.search(r"[0-9a-fA-F]{8,}", name):
        name = re.sub(r"\s+", "", name)
    else:
        name = re.sub(r"\s+", " ", name)
        
    return name.strip()

def clean_normalized_label(label: str) -> str:
    """
    ลบช่องว่างที่ซ้ำกันตั้งแต่ 2 ช่องขึ้นไปให้เหลือช่องว่างเดียว
    """
    return re.sub(r"\s{2,}", " ", label).strip()

# --- Extraction Helper Functions ---

def normalize_filename(name: str) -> str:
    """
    ทำความสะอาดชื่อไฟล์: ใช้ Regex เพื่อดึงชื่อไฟล์ที่ถูกต้องที่มีนามสกุล 
    """
    # Look for a file pattern with extension
    match = re.search(r"([^/\\\s\(\[\{]*\.(?:pdf|xlsx|jpg|png|docx|pptx|zip))", name, re.IGNORECASE)
    if not match:
        return ""
        
    base_name = match.group(1).strip()
    
    # Handle UUIDs with spaces 
    if re.match(r"^[0-9a-fA-F\-]{10,}", base_name.replace(' ', '')):
        base_name = base_name.replace(' ', '')

    return base_name if 3 <= len(base_name) < 120 else ""

def get_text_safe(page: fitz.Page) -> str:
    """Extract clean text from a PDF page, handling corrupted characters and normalizing whitespace."""
    try:
        text = page.get_text("text")
        if "\x00" in text:
            raise ValueError("Found null characters")
    except Exception:
        text = " ".join(block[4] for block in page.get_text("blocks"))
        
    return re.sub(r"\s+", " ", text).strip()

# --- Main Extraction Function ---

def extract_km_levels_with_description(pdf_path, organization="PEA", section="KM"):
    """
    Extracts KM levels (ระดับ 1–5), their main criteria, the client's description, 
    and links evidence files, handling descriptions that span multiple pages.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}. Returning empty data.")
        return []

    doc = fitz.open(pdf_path)
    results: List[Dict[str, Any]] = []

    # Defined KM structure for Category/Subtopic tracking
    KM_STRUCTURE = {
        "1. การนำองค์กร": ["1.1 วิสัยทัศน์", "1.2 การมีส่วนร่วมของผู้บริหาร"],
        "2. การวางแผนและทรัพยากรสนับสนุน": ["2.1 การวางแผนและทรัพยากรสนับสนุน", "2.2 การจัดสรรทรัพยากร"],
        "3. บุคลากร": [
            "3.1 ความตระหนักความเข้าใจการมีส่วนร่วม และการสร้างแรงจูงใจด้าน KM",
            "3.2 วัฒนธรรมและสภาพแวดล้อมการทำงาน",
            "3.3 ความสามารถและความรับผิดชอบของทีมงาน",
        ],
        "4. กระบวนการจัดการความรู้": [
            "4.1 กระบวนการจัดการความรู้ที่เป็นระบบและมีการนำเทคโนโลยีมาประยุกต์ใช้",
            "4.2 สารสนเทศ/ความรู้จากหน่วยงานภายนอก",
        ],
        "5. กระบวนการปฏิบัติ": [
            "5.1 การปฏิบัติงานโดยใช้ความรู้เป็นฐาน",
            "5.2 การสร้างความตระหนักเรื่องความเสี่ยงโดยการใช้ความรู้เป็นฐาน",
        ],
        "6. ผลลัพธ์ของการจัดการความรู้": ["6.1 ผลงานด้านต่างๆที่เกี่ยวข้องกับการจัดการความรู้"],
    }
    
    FILE_PATTERN = re.compile(r"([^\s/\\\"']+\.(?:pdf|xlsx|jpg|png|docx|pptx|zip))", flags=re.IGNORECASE)
    SUB_ID_PATTERN = re.compile(r"(\d\.\d)")
    
    # Pattern for cutting off numbered evidence lists at the end of the description
    CITATION_START_PATTERN = re.compile(r"\s*\d+\.\s+", flags=re.DOTALL) 
    
    # Pattern for cleaning up PDF artifacts (dates, URLs, page numbers, and "คำอธิบาย")
    HEADER_CLEANUP_PATTERN = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{4}.*?)\s*(คำอธิบาย)?[:：]?" # Date/URL/Page, followed by optional "คำอธิบาย"
        r"|(\w+\.center\.sepo\.go\.th.*?)" # Specific URL pattern
        r"|(คำอธิบาย[:：]?)", # Just "คำอธิบาย" alone
        flags=re.IGNORECASE | re.DOTALL
    )

    # State tracking for cross-page extraction
    current_category: str | None = None
    current_subtopic: str | None = None
    pending_evidence: Set[str] = set()
    pending_entry: Dict[str, Any] | None = None

    def clean_and_commit(entry: Dict[str, Any], results: List[Dict[str, Any]]):
        """Applies final description cleaning and commits the entry."""
        if not entry:
            return

        desc_part = entry.get("level_description", "")
        
        # 1. Clean PDF artifacts/Headers
        desc_part = HEADER_CLEANUP_PATTERN.sub("", desc_part).strip()

        # 2. Remove leading bullet points/dashes from description
        desc_part = re.sub(r"^\s*[-*]+\s*", "", desc_part).strip()

        # 3. Truncate citation/numbered list part (which often appears at the end)
        citation_match = CITATION_START_PATTERN.search(desc_part)
        if citation_match:
            truncation_index = citation_match.start()
            desc_part = desc_part[:truncation_index]
        
        # 4. Final text cleanup
        desc_part = desc_part.strip()
        desc_part = re.sub(r"\s+", " ", desc_part).strip()
        desc_part = re.sub(r"\s*-\s*", "- ", desc_part)
        desc_part = re.sub(r"^\s*คำอธิบาย\s*[:：]?", "", desc_part).strip()
        
        entry["level_description"] = desc_part
        
        results.append(entry)


    for page in doc:
        text = get_text_safe(page)
        page_num = page.number + 1
        sub_criteria_id: str | None = None

        # 1. Update Category/Subtopic based on text on current page
        for cat, subs in KM_STRUCTURE.items():
            if cat in text:
                current_category = cat
            for sub in subs:
                id_match = SUB_ID_PATTERN.search(sub)
                if id_match:
                    found_sub_criteria_id = id_match.group(1)
                    
                    if sub.replace(' ', '').lower() in text.replace(' ', '').lower(): 
                        current_subtopic = sub
                        sub_criteria_id = found_sub_criteria_id
                        break 
        
        # Use current subtopic ID if it wasn't explicitly found on this page but is still active
        if not sub_criteria_id and current_subtopic:
             id_match = SUB_ID_PATTERN.search(current_subtopic)
             if id_match:
                sub_criteria_id = id_match.group(1)
                
        # 2. Extract Files on this page and add to pending set
        raw_files = FILE_PATTERN.findall(text)
        clean_files = {normalize_filename(f) for f in raw_files}
        pending_evidence.update(clean_files - {""})

        # 3. Find Level Blocks (Start of new entry)
        # Using strict regex to only match Levels 1-5
        level_blocks = re.findall(
            r"(ระดับ\s*[1-5][\s\S]*?)(?=ระดับ\s*[1-5]|$)", text, flags=re.DOTALL
        )
        
        # If no new level is found, check for continuation
        if not level_blocks and pending_entry:
            pending_entry["level_description"] += " " + text
            continue 

        # If a new level is found, commit the previous pending one if it exists
        if level_blocks and pending_entry:
            pending_entry["evidence_files"] = sorted(list(pending_evidence))
            pending_entry["evidence_count"] = len(pending_entry["evidence_files"])
            pending_entry["valid_evidence_count"] = pending_entry["evidence_count"]
            clean_and_commit(pending_entry, results)
            pending_entry = None
            pending_evidence.clear()

        # Process the found blocks on this page
        for i, block in enumerate(level_blocks):
            level_match = re.match(r"ระดับ\s*(\d+)", block)
            if not level_match:
                continue

            lvl = int(level_match.group(1))
            
            # Use 'คำอธิบาย' as the primary separator. If missing, desc_part is empty.
            split_match = re.split(r"คำอธิบาย[:：]?", block, maxsplit=1)
            
            main_part = split_match[0]
            desc_part = split_match[1] if len(split_match) == 2 else "" # Empty if 'คำอธิบาย' is missing

            # Clean main part (remove "ระดับ X" from the front)
            main_part = re.sub(r"ระดับ\s*\d+\s*", "", main_part).strip()

            new_entry = {
                "organization": organization,
                "section": section,
                "category": current_category,
                "subtopic": current_subtopic,
                "sub_criteria_id": sub_criteria_id, 
                "level": f"Level {lvl}",
                "level_main": main_part,
                "level_description": desc_part, 
                "page": page_num
            }

            # If it's the last block on the page, keep it pending for continuation
            if i == len(level_blocks) - 1 and page.number < doc.page_count - 1:
                pending_entry = new_entry
            else:
                new_entry["evidence_files"] = sorted(list(pending_evidence))
                new_entry["evidence_count"] = len(pending_evidence)
                new_entry["valid_evidence_count"] = new_entry["evidence_count"]
                clean_and_commit(new_entry, results)
                pending_evidence.clear()

    # 4. Final Commit after loop
    if pending_entry:
        pending_entry["evidence_files"] = sorted(list(pending_evidence))
        pending_entry["evidence_count"] = len(pending_entry["evidence_files"])
        pending_entry["valid_evidence_count"] = pending_entry["evidence_count"]
        clean_and_commit(pending_entry, results)
        
    return results

# --- Main Execution Block (Extraction + In-Memory Cleaning + Single Save) ---

if __name__ == "__main__":
    pdf_path = os.path.join("evidence_checklist", "km_evidence_description.pdf")
    
    # 1. Extraction Phase: Get raw data
    raw_data = extract_km_levels_with_description(pdf_path)

    os.makedirs("output", exist_ok=True)
    cleaned_path = "output/km_levels_cleaned.json" # Final output file

    # 2. Final Cleaning Phase: Process data in memory
    cleaned_data = []
    
    for entry in raw_data:
        entry["normalized_label"] = clean_normalized_label(entry.get("normalized_label", ""))

        # B. Clean and filter filenames
        cleaned = [clean_filename(f) for f in entry["evidence_files"]]
        
        # Filter out UUID-like files and empty strings
        filtered = [f for f in cleaned if f and not is_uuid_like(f)]

        # C. Update entry
        entry["evidence_files"] = sorted(list(set(filtered)))
        entry["valid_evidence_count"] = len(entry["evidence_files"])
        entry["evidence_count"] = entry["valid_evidence_count"]
        
        # D. Remove unnecessary/empty fields
        if "normalized_label" in entry and not entry["normalized_label"]:
             del entry["normalized_label"]
        
        cleaned_data.append(entry)

    # 3. Single Save Phase: Save the final cleaned data
    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Extracted and Cleaned JSON saved → {cleaned_path}")
    print(f"Total entries processed: {len(cleaned_data)}")
    
    # Display sample of the final output
    print("\n--- ตัวอย่าง 3 รายการแรกของผลลัพธ์ (ข้อมูลที่ล้างแล้ว) ---")
    print(json.dumps(cleaned_data[:3], indent=2, ensure_ascii=False))
