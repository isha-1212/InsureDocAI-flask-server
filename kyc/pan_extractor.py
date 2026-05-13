import requests
from PIL import Image
import pytesseract
import re
import cv2
import numpy as np
from io import BytesIO
import os


def extract_pan_info_from_url(image_url, debug=False):
    try:
        # Handle both URLs and local file paths
        if image_url.startswith('http'):
            # Download image from URL
            response = requests.get(image_url)
            image_file = BytesIO(response.content)
            img = Image.open(image_file)
        else:
            # Load local file
            if not os.path.exists(image_url):
                return {"error": f"File not found: {image_url}"}
            img = Image.open(image_url)
        
        # Read & preprocess with simpler approach
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Try simple upscaling first (minimal preprocessing)
        gray_upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Optional: Light denoising
        denoised = cv2.fastNlMeansDenoising(gray_upscaled, None, 10, 7, 21)
        
        # Optional: Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        img_final = Image.fromarray(enhanced)

    except Exception as e:
        return {"error": f"Image processing failed: {str(e)}"}

    # Try multiple OCR approaches
    text_results = []
    
    # Approach 1: Enhanced image with PSM 6
    text1 = pytesseract.image_to_string(img_final, config=r"--oem 3 --psm 6")
    text_results.append(text1)
    
    # Approach 2: Enhanced image with PSM 3 (fully automatic)
    text2 = pytesseract.image_to_string(img_final, config=r"--oem 3 --psm 3")
    text_results.append(text2)
    
    # Approach 3: Original image with minimal processing
    img_original_upscaled = cv2.resize(np.array(img.convert('RGB')), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img_orig = Image.fromarray(img_original_upscaled)
    text3 = pytesseract.image_to_string(img_orig, config=r"--oem 3 --psm 6")
    text_results.append(text3)
    
    # Combine all text results
    text = "\n".join(text_results)
    
    # Debug mode: print all extracted text
    if debug:
        print("\n" + "="*60)
        print("DEBUG: RAW OCR TEXT FROM PAN CARD")
        print("="*60)
        for i, t in enumerate(text_results, 1):
            print(f"\n--- Approach {i} ---")
            print(t[:200] if len(t) > 200 else t)
        print("="*60 + "\n")
    
    # More relaxed keyword check (optional - just for warning)
    pan_keywords = [
        "income", "tax", "govt", "government",
        "permanent", "account", "number", "department", "india", "pan", "card"
    ]
    
    has_keywords = any(k in text.lower() for k in pan_keywords)
    
    # Try to extract PAN number with more flexible pattern
    # PAN format: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)
    pan_patterns = [
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",  # Standard format
        r"[A-Z]{5}\s?[0-9]{4}\s?[A-Z]",  # With possible spaces
    ]
    
    found_pan = None
    for pattern in pan_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Clean and validate
            for match in matches:
                clean_pan = match.replace(" ", "")
                if len(clean_pan) == 10 and clean_pan[:5].isalpha() and clean_pan[5:9].isdigit() and clean_pan[9].isalpha():
                    found_pan = clean_pan
                    break
        if found_pan:
            break

    if found_pan:
        return {
            "pan_number": found_pan,
            "message": "PAN number successfully extracted",
            "keyword_warning": "PAN keywords not found in image - please verify" if not has_keywords else None
        }
    
    # If still nothing found, return informative error
    return {
        "error": "PAN number not found - image quality may be too low for OCR",
        "suggestion": "Try using a higher resolution image or ensure the PAN card is clearly visible"
    }
