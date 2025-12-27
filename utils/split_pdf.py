import PyPDF2
import argparse
import os
import sys

def split_pdf(input_path, start_page, end_page):
    # 1. ตรวจสอบว่าไฟล์ต้นฉบับมีอยู่จริงหรือไม่
    if not os.path.exists(input_path):
        print(f"Error: ไม่พบไฟล์ที่ '{input_path}'")
        return

    try:
        # --- เพิ่มส่วนการจัดการโฟลเดอร์ exports ---
        output_dir = "exports"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 สร้างโฟลเดอร์ '{output_dir}' เรียบร้อยแล้ว")
        # --------------------------------------

        with open(input_path, "rb") as file_in:
            reader = PyPDF2.PdfReader(file_in)
            total_pages = len(reader.pages)

            if start_page < 1 or end_page > total_pages or start_page > end_page:
                print(f"Error: ช่วงหน้าไม่ถูกต้อง! (ไฟล์มี {total_pages} หน้า)")
                return

            writer = PyPDF2.PdfWriter()
            for page_num in range(start_page - 1, end_page):
                writer.add_page(reader.pages[page_num])

            # 2. ตั้งชื่อไฟล์และกำหนดให้อยู่ในโฟลเดอร์ exports
            base_filename = os.path.splitext(os.path.basename(input_path))[0]
            output_filename = f"{base_filename}_extracted_{start_page}_to_{end_page}.pdf"
            
            # รวม path ให้เป็น exports/ชื่อไฟล์.pdf
            output_path = os.path.join(output_dir, output_filename)

            # 3. บันทึกไฟล์
            with open(output_path, "wb") as file_out:
                writer.write(file_out)

            print("-" * 30)
            print(f"✅ แบ่งไฟล์สำเร็จ!")
            print(f"📍 บันทึกไปที่: {output_path}")
            print(f"📄 รวมทั้งหมด: {end_page - start_page + 1} หน้า")
            print("-" * 30)

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

def main():
    parser = argparse.ArgumentParser(description="สคริปต์แบ่งไฟล์ PDF และเก็บในโฟลเดอร์ exports")
    parser.add_argument("path", help="Path ไปยังไฟล์ PDF ต้นฉบับ")
    parser.add_argument("start", type=int, help="หน้าเริ่มต้น")
    parser.add_argument("to", type=int, help="หน้าสิ้นสุด")

    args = parser.parse_args()
    split_pdf(args.path, args.start, args.to)

if __name__ == "__main__":
    main()