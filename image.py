import pytesseract
from PIL import Image
from tkinter import Tk, filedialog

# ✅ Tell Python where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === File upload dialog ===
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
)

if image_path:
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img, lang="eng")

    print("\n✅ Extracted Text:\n")
    print(extracted_text)

    with open("output_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print("\n📄 Text saved in 'output_text.txt'")
else:
    print("❌ No image selected.")
