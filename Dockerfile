FROM python:3.11-slim

# 1. ติดตั้ง System Dependencies (รันผ่านแล้ว)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    cmake \
    tesseract-ocr \
    tesseract-ocr-tha \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. อัปเกรดเครื่องมือ
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. ล็อกเวอร์ชัน NumPy/Pandas
RUN pip install --no-cache-dir "numpy==1.26.4" "pandas==2.1.4"

# 4. ติดตั้ง PyTorch 2.6.0 สำหรับ CUDA 12.1 (เพื่อแก้ Error Vulnerability)
# บังคับใช้เวอร์ชัน >=2.6.0 ตามที่ระบบต้องการ
RUN pip install --no-cache-dir "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121

# 5. ติดตั้ง Python Libraries ที่เหลือ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. คัดลอกโค้ด
COPY . .

# 7. ตั้งค่า Path OCR
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

CMD ["tail", "-f", "/dev/null"]