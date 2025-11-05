from pypdf import PdfReader, PdfWriter
import os

# -------------------------------
# Config
# -------------------------------
input_pdf_path = "data/seam/SE-AM Manual Book 2566 ฉบับสมบูรณ์.pdf"
output_folder = "data/seam"
SUPPORTED_ENABLERS = ["CG", "SP", "RM&IC", "SM", "CM", "DT", "HCM", "KM", "IM", "IA"]

# ตัวอย่าง Mapping หน้า PDF ต่อ Enabler (ปรับตามเอกสารจริง)
ENABLER_PAGE_RANGES = {
    "CG": (4, 30),        # หน้า 5-31
    "SP": (31, 59),       # หน้า 32-60
    "RM&IC": (60, 89),    # หน้า 61-90
    "SM": (90, 106),      # หน้า 91-107 module 1 stakeholder
    "CM": (107, 118),      # หน้า 108-119 module 2 customer
    "DT": (119, 200),     # หน้า 120-201
    "HCM": (201, 231),    # หน้า 202-232
    "KM": (232, 250),     # หน้า 233-251
    "IM": (251, 265),     # หน้า 252-266
    "IA": (266, 287)      # หน้า 267-288
}

# -------------------------------
# Load PDF
# -------------------------------
reader = PdfReader(input_pdf_path)

# -------------------------------
# Export per Enabler
# -------------------------------
for enabler in SUPPORTED_ENABLERS:
    if enabler not in ENABLER_PAGE_RANGES:
        print(f"⚠️ Skipping {enabler}, no page range defined.")
        continue

    start_idx, end_idx = ENABLER_PAGE_RANGES[enabler]
    writer = PdfWriter()

    for i in range(start_idx, end_idx + 1):
        writer.add_page(reader.pages[i])

    output_path = os.path.join(output_folder, f"seam_rubrics_{enabler}.pdf")
    with open(output_path, "wb") as f_out:
        writer.write(f_out)

    print(f"✅ Exported {enabler}: pages {start_idx+1}-{end_idx+1} -> {output_path}")
