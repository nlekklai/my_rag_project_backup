from PIL import Image
import pytesseract

img_path = "thai_sample.png"  # replace with your image
text = pytesseract.image_to_string(img_path, lang='tha+eng')
print(text)
