import json
import re

def is_uuid_like(name: str) -> bool:
    """Check if filename mostly UUID pattern (avoid false positive)"""
    return bool(re.fullmatch(r"[0-9a-fA-F\-]{20,}\.(pdf|jpg|png|docx|xlsx)", name.strip()))

def clean_filename(name: str) -> str:
    """Merge UUID splits and normalize whitespace"""
    name = re.sub(r"\s+", "", name) if re.search(r"[0-9a-fA-F]{8,}", name) else re.sub(r"\s+", " ", name)
    return name.strip()

def clean_normalized_label(label: str) -> str:
    """Remove double spaces and align format"""
    return re.sub(r"\s{2,}", " ", label).strip()

def clean_json(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        # Normalize label
        entry["normalized_label"] = clean_normalized_label(entry.get("normalized_label", ""))

        # Clean filenames
        cleaned = [clean_filename(f) for f in entry["evidence_files"]]
        filtered = [f for f in cleaned if not is_uuid_like(f)]

        entry["evidence_files"] = sorted(set(filtered))
        entry["valid_evidence_count"] = len(entry["evidence_files"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Cleaned JSON saved → {output_path}")

if __name__ == "__main__":
    clean_json("output/mappings_pea_by_category.json", "output/mappings_pea_v3_clean.json")
