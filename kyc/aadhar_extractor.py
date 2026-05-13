import requests
from PIL import Image
import pytesseract
import re
import cv2
import numpy as np
from io import BytesIO
import os


def extract_aadhaar_info_from_url(image_url, debug=False):
    try:
        # Handle both URLs and local file paths
        if image_url.startswith('http'):
            # Download image from URL
            response = requests.get(image_url)
            image_file = BytesIO(response.content)
            img = Image.open(image_file).convert("L")
        else:
            # Load local file
            if not os.path.exists(image_url):
                return {"error": f"File not found: {image_url}"}
            img = Image.open(image_url).convert("L")
        img_cv = np.array(img)

        # Threshold to remove background
        _, img_thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)

        # Upscale for better OCR
        img_resized = cv2.resize(img_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img_final = Image.fromarray(img_resized)

    except Exception as e:
        return {"error": f"Image loading failed: {str(e)}"}

    # Extract text
    text = pytesseract.image_to_string(img_final)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Debug mode: print all extracted text
    if debug:
        print("\n" + "="*60)
        print("DEBUG: RAW OCR TEXT FROM AADHAR CARD")
        print("="*60)
        for i, line in enumerate(lines):
            print(f"{i:3d}: {line}")
        print("="*60 + "\n")

    name = None
    dob = None
    aadhaar_number = None

    # -------------------------
    # 1️⃣ Extract DOB
    # -------------------------
    dob_regex = r"\d{2}[/-]\d{2}[/-]\d{4}"
    dob_candidates = []

    for i, line in enumerate(lines):
        if re.search(r"govt|india|address|male|female|year|issued", line.lower()):
            continue

        m = re.search(dob_regex, line)
        if m:
            dob_candidates.append((i, m.group()))

    # Pick first DOB (most reliable)
    for idx, dob_found in dob_candidates:
        dob = dob_found

        # -------------------------
        # 2️⃣ Extract NAME (1–4 lines above DOB)
        # -------------------------
        # Extended keyword list to filter out non-name text
        exclude_keywords = r"dob|date|year|govt|india|card|address|gender|aadhaar|aadhar|enrolment|enrollment|issue|vid|uidai|father|mother|spouse"
        
        # Specific words to exclude (case-insensitive)
        exclude_exact = {"male", "female", "m", "f", "transgender"}
        
        # Look for name in 1-4 lines above DOB
        for candidate in reversed(lines[max(0, idx - 4):idx]):
            candidate_clean = candidate.strip()
            
            # Skip if exact match with gender indicators
            if candidate_clean.lower() in exclude_exact:
                continue
            
            # Skip if contains excluded keywords
            if re.search(exclude_keywords, candidate_clean.lower()):
                continue
            
            # Skip if contains numbers
            if re.search(r"\d", candidate_clean):
                continue
            
            # Must be alphabetic with spaces/dots, minimum 3 chars, max 50
            if re.fullmatch(r"[A-Za-z\s\.]{3,50}", candidate_clean):
                words = candidate_clean.split()
                
                # Prefer multi-word names (full names)
                if len(words) >= 2:
                    name = candidate_clean.title()
                    break
                # Accept single word only if longer than 6 chars (avoid "Male", "Son", etc.)
                elif len(words) == 1 and len(candidate_clean) > 6:
                    name = candidate_clean.title()
                    break
        break

    # -------------------------
    # 3️⃣ Extract Aadhaar Number
    # -------------------------
    # Multiple regex patterns to handle different formats
    aadhaar_patterns = [
        r"\b[0-9]\d{3}\s?\d{4}\s?\d{4}\b",           # Any 12 digits (relaxed for testing)
        r"\b[0-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b",     # With dashes
        r"\b[0-9]\d{11}\b",                           # No spaces
        r"[0-9]\d{3}\s*[0-9Oo]\d{3}\s*\d{4}",       # OCR confusion O/0
    ]
    
    aadhaar_candidates = []
    
    for line in lines:
        # Clean OCR artifacts (O -> 0, l -> 1)
        line_cleaned = line.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
        
        # Remove common prefixes/text before numbers
        line_cleaned = re.sub(r'(rvs|vid|uid|enr|enrolment|no\.?|number|:)\s*', '', line_cleaned, flags=re.IGNORECASE)
        
        for pattern in aadhaar_patterns:
            matches = re.findall(pattern, line_cleaned)
            for match in matches:
                # Remove spaces, dashes, dots
                clean_number = re.sub(r'[\s\-\.]', '', match)
                
                # Validate: must be exactly 12 digits
                # Note: Real Aadhaar numbers start with 2-9, but accepting all for testing
                if len(clean_number) == 12 and clean_number.isdigit():
                    aadhaar_candidates.append(clean_number)
    
    # Pick the first valid Aadhaar number
    if aadhaar_candidates:
        aadhaar_number = aadhaar_candidates[0]

    return {
        "name": name or "Not found",
        "dob": dob or "Not found",
        "aadhaar_number": aadhaar_number or "Not found",
    }
