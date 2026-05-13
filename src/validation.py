import re
import math
from typing import List, Dict, Optional

# ================= SIGMOID NORMALIZATION =================

def sigmoid_normalize(score: float) -> float:
    """Normalize score to 0-1 range using sigmoid function"""
    return 1 / (1 + math.exp(-score))

# ================= CONFIG =================

GENERIC_HEADERS = {
    "invoice", "bill", "receipt", "tax invoice",
    "cash memo", "summary", "statement", "bill summary",
    "invoice summary", "payment summary", "final bill",
    "patient details", "patient info", "patient information",
    "details", "information", "account statement",
    "patient account statement", "account summary"
}

TABLE_HEADERS = {
    "item name", "description", "qty", "quantity",
    "rate", "price", "amount", "mrp", "total",
    "emergency", "icu", "ot", "opd", "ipd", "ward"
}

ADDRESS_KEYWORDS = {
    "road", "rd", "street", "st", "lane", "ln",
    "sector", "nagar", "area", "city",
    "pune", "mumbai", "delhi", "india"
}

# Hospital-specific keywords that boost confidence
HOSPITAL_KEYWORDS = {
    "hospital", "clinic", "medical", "health", "care",
    "診所", "醫院", "pharmacy", "diagnostic", "centre",
    "center", "nursing", "home", "polyclinic", "multispeciality",
    "multispecialty", "speciality", "specialty", "apollo",
    "fortis", "medanta", "aiims", "max", "medplus", "pharma",
    "drugstore", "medicos", "medicals"
}

CONF_HIGH = 0.5  # Lowered from 0.6 to be more lenient
CONF_LOW = 0.3   # between LOW–HIGH → accept but flag

# =========================================


def contains_hospital_keyword(text: str) -> bool:
    """Check if text contains common hospital-related keywords"""
    t = text.lower()
    return any(keyword in t for keyword in HOSPITAL_KEYWORDS)


def looks_like_address(text: str) -> bool:
    t = text.lower()
    if re.search(r"\b\d{6}\b", t):   # Indian pincode
        return True
    if any(k in t for k in ADDRESS_KEYWORDS):
        return True
    return False


def looks_like_table_header(text: str) -> bool:
    t = text.lower().strip()
    return t in TABLE_HEADERS


def looks_like_generic_header(text: str) -> bool:
    t = text.lower().strip()
    return t in GENERIC_HEADERS


def valid_hospital_name(text: str) -> bool:
    """Structural + negative validation - less strict to capture more candidates"""
    t = text.lower().strip()
    if len(t) < 2: return False
    if ':' in text: return False
    if re.match(r'^thank\s+you|^thanks', t, re.IGNORECASE): return False
    if re.search(r'thank\s+you\s+for\s+choosing|we\s+appreciate|come\s+again|visit\s+again', t, re.IGNORECASE): return False
    if looks_like_generic_header(t): return False
    if looks_like_table_header(t): return False
    if re.match(r'^\d+\s', text): return False
    if re.search(r'\bstate\s+\d{5}', t): return False
    if contains_hospital_keyword(text): return True
    if looks_like_address(t): return False
    digit_count = sum(c.isdigit() for c in t)
    if digit_count >= 6: return False
    if t.isdigit(): return False
    return True


def hospital_name_score(text: str, confidence: float) -> float:
    """
    Final score = ML confidence × semantic_weight × heuristic_weight
    Industry-style re-ranking, not just filtering
    """
    score = confidence
    heuristic_weight = 1.0
    
    text_lower = text.lower().strip()
    word_count = len(text.split())
    
    # ===== STEP 1: Penalize TAGLINES & PROMOTIONAL TEXT =====
    promotional_words = ['your', 'best', 'partner', 'welcome', 'trusted', 
                         'quality', 'caring', 'we', 'our']
    if any(word in text_lower for word in promotional_words):
        heuristic_weight *= 0.3  # Heavy penalty for taglines
    
    # ===== STEP 2: Penalize GENERIC HEADERS =====
    generic_headers = ['patient information', 'hospital information', 
                      'payment mode', 'bill details', 'invoice details']
    if text_lower in generic_headers:
        heuristic_weight *= 0.1  # Massive penalty for headers
    
    # ALL CAPS + Generic = likely header
    if text.isupper() and word_count <= 2:
        generic_single_words = ['information', 'details', 'payment', 'mode', 
                               'patient', 'bill', 'invoice', 'receipt']
        if any(word in text_lower for word in generic_single_words):
            heuristic_weight *= 0.1
    
    # ===== STEP 3: BOOST ACTUAL BUSINESS NAMES =====
    # Strong indicators of real business name
    business_keywords = ['drugstore', 'pharmacy', 'clinic', 'hospital', 
                        'medical center', 'healthcare center', 'polyclinic', 
                        'medicos', 'medicals', 'nursing home']
    
    # Check if it's ONLY a generic word without a proper business name
    standalone_generic = ['pharmacy', 'hospital', 'clinic', 'medical', 'drugstore', 
                         'medicos', 'medicals', 'chemist', 'healthcare']
    if text_lower in standalone_generic or (word_count == 1 and any(text_lower == word for word in standalone_generic)):
        heuristic_weight *= 0.2  # Heavy penalty for standalone generic words
    elif any(keyword in text_lower for keyword in business_keywords):
        heuristic_weight *= 1.8  # Strong boost for actual business names
    
    # ===== STEP 4: LENGTH-BASED SCORING =====
    # Real hospital names: 2-6 words
    if 2 <= word_count <= 6:
        heuristic_weight *= 1.2
    elif word_count > 6:
        heuristic_weight *= 0.7  # Too long, likely description
    elif word_count == 1:
        heuristic_weight *= 0.5  # Single word, likely partial or generic
    
    # Character length check
    text_len = len(text.strip())
    if 10 <= text_len <= 40:
        heuristic_weight *= 1.1  # Good length for business name
    elif text_len < 5:
        heuristic_weight *= 0.5  # Too short
    elif text_len > 50:
        heuristic_weight *= 0.6  # Too long
    
    # ===== FINAL SCORE =====
    final_score = score * heuristic_weight
    
    return final_score


# ================= MAIN VALIDATION =================

def validate_hospital_name(
    ml_predictions: List[Dict]
) -> Dict:
    """
    Input:
        ml_predictions = [
          {"text": "...", "label": "hospital_name", "confidence": 0.82},
          ...
        ]

    Output:
        {
          "hospital_name": str or None,
          "confidence": float or None,
          "status": "accepted" | "accepted_low_confidence" | "rejected",
          "reason": str
        }
    """

    candidates = []

    # STEP 1: Collect all hospital_name labeled items with their positions
    hospital_name_items = []
    for idx, item in enumerate(ml_predictions):
        if item["label"] != "hospital_name":
            continue
        
        text = item["text"].strip()
        confidence = float(item["confidence"])
        hospital_name_items.append((idx, text, confidence))
    
    # STEP 2: Try to merge consecutive hospital_name tokens (but stop at taglines/headers)
    merged_candidates = []
    i = 0
    while i < len(hospital_name_items):
        # Start a potential merge group
        merge_group = [hospital_name_items[i]]
        j = i + 1
        
        # Look ahead for consecutive tokens (within 2 positions)
        while j < len(hospital_name_items):
            current_idx = merge_group[-1][0]
            next_idx = hospital_name_items[j][0]
            next_text = hospital_name_items[j][1].lower()
            
            # Stop merging if we hit promotional words, headers, or document types
            stop_words = ['your', 'our', 'we', 'customer', 'patient', 'information', 
                         'details', 'welcome', 'thank', 'best', 'quality',
                         'invoice', 'bill', 'receipt', 'statement', 'summary']
            if any(word in next_text for word in stop_words):
                break
            
            # If next token is within 2 positions, merge it
            if next_idx - current_idx <= 2:
                merge_group.append(hospital_name_items[j])
                j += 1
            else:
                break
        
        # Create merged text
        merged_text = " ".join([item[1] for item in merge_group])
        
        # Remove trailing document type words (invoice, bill, receipt, etc.)
        doc_type_words = ['invoice', 'bill', 'receipt', 'statement', 'summary']
        words = merged_text.split()
        while words and words[-1].lower() in doc_type_words:
            words.pop()
        merged_text = " ".join(words)
        
        # Skip if nothing remains after filtering
        if not merged_text.strip():
            i = j
            continue
        
        avg_confidence = sum([item[2] for item in merge_group]) / len(merge_group)
        
        # If we have multiple tokens merged, boost the confidence
        if len(merge_group) > 1:
            # Boost confidence for merged multi-word names
            avg_confidence *= 1.2
        
        merged_candidates.append((merged_text, avg_confidence, len(merge_group)))
        i = j

    # STEP 3: Score all merged candidates
    for text, confidence, token_count in merged_candidates:
        if not valid_hospital_name(text):
            continue
        
        # Calculate heuristic-weighted score
        score = hospital_name_score(text, confidence)
        
        # Extra boost for multi-word names (they're more likely to be complete business names)
        if token_count > 1:
            score *= 1.5
        
        candidates.append((text, score))

    # Fallback: search all predictions for long text with hospital keywords
    # (in case full name isn't labeled as hospital_name)
    if not candidates or (candidates and max(c[1] for c in candidates) < 1.5):
        for item in ml_predictions:
            text = item["text"].strip()
            confidence = float(item["confidence"])
            
            # Look for longer text with hospital keywords
            if len(text) > 15 and contains_hospital_keyword(text) and valid_hospital_name(text):
                score = hospital_name_score(text, confidence * 0.8)  # Reduce confidence slightly
                candidates.append((text, score))

    if not candidates:
        return {
            "hospital_name": None,
            "confidence": None,
            "status": "rejected",
            "reason": "no_valid_hospital_name_candidate"
        }

    # Pick best candidate
    best_text, best_score = max(candidates, key=lambda x: x[1])

    # Normalize score using sigmoid
    normalized_score = sigmoid_normalize(best_score)
    
    if best_score >= CONF_HIGH:
        return {
            "hospital_name": best_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted",
            "reason": "high_confidence_structural_match"
        }

    if CONF_LOW <= best_score < CONF_HIGH:
        return {
            "hospital_name": best_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted_low_confidence",
            "reason": "low_confidence_but_structurally_valid"
        }

    return {
        "hospital_name": None,
        "confidence": None,
        "status": "rejected",
        "reason": "confidence_too_low"
    }


# ================= PATIENT NAME VALIDATION =================

def valid_patient_name(text: str) -> bool:
    """Validate patient name structure"""
    t = text.lower().strip()
    
    if len(t) < 2:
        return False
    
    # Reject common label/header patterns (including misspellings)
    header_patterns = [
        r'patient\s*information',
        r'palient\s*information',
        r'patient\s*details',
        r'patient\s*info',
        r'patient\s*issue',
        r'name\s*information',
        r'customer\s*information',
        r'guardian\s*name',
        r'admit\s*date',
        r'room\s*category'
    ]
    for pattern in header_patterns:
        if re.search(pattern, t):
            return False
    
    # Reject if it contains hospital/pharmacy keywords (patient name can't be a hospital)
    if any(keyword in t for keyword in HOSPITAL_KEYWORDS):
        return False
    
    # Reject table column headers
    table_headers = ['item', 'qty', 'quantity', 'price', 'rate', 'amount', 'gst', 'total', 'description', 'subtotal', 'product', 'products', 'sr', 'no', 'sno', 's.no', 'particulars', 'product name', 'item name']
    if t in table_headers:
        return False
    
    # Reject single uppercase words that are likely headers
    if len(text.split()) == 1 and text.isupper() and len(text) < 15:
        common_headers = ['product', 'item', 'name', 'details', 'description', 'qty', 'rate', 'amount', 'total', 'date', 'time']
        if t in common_headers:
            return False
    
    # Reject "Product Name" or "Item Name" specifically (case insensitive)
    if re.match(r'^(product|item)\s+name$', t, re.IGNORECASE):
        return False
    
    # Reject shipping/delivery terms
    shipping_terms = ['ship to', 'ship', 'deliver to', 'delivery', 'bill to', 'sold to']
    if any(term in t for term in shipping_terms):
        return False
    
    # Reject product descriptions and quantities (medicine names)
    product_patterns = [r'\d+\s*ml', r'\d+\s*mg', r'\d+\s*gm', r'\d+\s*kg', 
                       r'\d+\s*tab', r'\d+\s*cap', r'\d+\s*bottle',
                       r'\bliq\b', r'\bsyrup\b', r'\btablet\b', r'\bcapsule\b',
                       r'strip\s+of\s+\d+', r'\(strip', r'amoxicillin', r'paracetamol',
                       r'aspirin', r'ibuprofen', r'\bmedicine\b', r'\bdrug\b']
    for pattern in product_patterns:
        if re.search(pattern, t):
            return False
    
    # Reject labels (text ending with colon like "Name:")
    if text.strip().endswith(':'):
        return False
    
    # Reject very short text (likely labels or incomplete)
    if len(text.strip()) < 3:
        return False
    
    # Reject single word without substantial length
    words = text.split()
    if len(words) == 1 and len(text) < 4:
        return False
    
    # Reject if contains label-related keywords ALONE (like "INVOICE DATE", "PATIENT NAME", etc.)
    # But allow names like "Mrs. Mary" that aren't labels
    label_only_patterns = [r'^name:?$', r'^date:?$', r'^invoice', r'^bill',
                          r'^receipt', r'^patient\s*(name|info|details)?:?$', 
                          r'^doctor', r'^hospital',
                          r'^account', r'^number', r'^no\.?$', r'^id:?$',
                          r'^charges?', r'^payment', r'^total', r'^amount',
                          r'^age[:/]', r'^gender[:/]', r'^sex[:/]',
                          r'ship\s*to:?$', r'bill\s*to:?$', r'sold\s*to:?$', r'deliver\s*to:?$']
    for pattern in label_only_patterns:
        if re.search(pattern, t):
            return False
    
    # Reject age/gender patterns like "Age/Gender:", "59/Female", "Age: 25"
    age_gender_patterns = [r'age[/:]', r'gender[/:]', r'\d+\s*/\s*(male|female)',
                          r'age.*gender', r'\d+\s*years?\s*(old)?',
                          r'(male|female)\s*/\s*\d+']
    for pattern in age_gender_patterns:
        if re.search(pattern, t):
            return False
    
    # Reject generic headers
    if looks_like_generic_header(t) or looks_like_table_header(t):
        return False
    
    # Reject if contains doctor/medical professional indicators
    doctor_patterns = [r'\bdr\.', r'\bdoctor\b', r'practitioner', r'surgeon', 
                      r'physician', r'consultant', r'specialist']
    for pattern in doctor_patterns:
        if re.search(pattern, t):
            return False
    
    # Reject if contains date-related keywords
    date_keywords = [r'discharge', r'admission', r'dob', r'birth',
                     r'\d{1,2}[/-]\w{3}[/-]\d{2}', r'\d{1,2}:\d{2}\s*[ap]m']
    for pattern in date_keywords:
        if re.search(pattern, t):
            return False
    
    # Reject if contains phone numbers (10 consecutive digits)
    if re.search(r'\d{10}', text):
        return False
    
    # Reject if contains common non-name patterns
    non_name_patterns = [r'reg\.?\s*no', r'mob\.?\s*no', r'ph:', r'mbbs', 
                         r'md', r'timing:', r'@', r'www\.', r'bed\s*no',
                         r'room\s*no', r'ward']
    for pattern in non_name_patterns:
        if re.search(pattern, t):
            return False
    
    # Reject if mostly digits (likely not a name)
    digit_count = sum(c.isdigit() for c in t)
    if digit_count > len(t) / 3:
        return False
    
    # Reject pure numbers
    if t.isdigit():
        return False
    
    # Reject addresses
    if looks_like_address(t):
        return False
    
    # Should have mostly alphabetic characters
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < len(text) * 0.5:
        return False
    
    return True


def patient_name_score(text: str, confidence: float) -> float:
    """Score patient name"""
    score = confidence
    
    # Title case is common for names
    if text.istitle():
        score += 0.15
    
    # Multi-word names
    word_count = len(text.split())
    if word_count >= 2:
        score += 0.1
    
    # Contains alphabets mostly
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count >= len(text) * 0.8:
        score += 0.1
    
    return min(score, 1.0)


def validate_patient_name(ml_predictions: List[Dict]) -> Dict:
    """Validate patient name from ML predictions - uses proximity to label indicators"""
    candidates = []
    
    # STEP 1: Find label positions (like "Patient Name:", "Name:")
    label_positions = []
    label_patterns = [r'patient\s*name', r'\bname\b', r'patient\s*info']
    
    for idx, item in enumerate(ml_predictions):
        text = item["text"].strip()
        # Check if this is a label indicator
        if any(re.search(pattern, text.lower()) for pattern in label_patterns):
            if ':' in text or text.lower() in ['name', 'patient name', 'patient']:
                label_positions.append(idx)
    
    # STEP 2: First pass - look for patient_name labeled items
    for idx, item in enumerate(ml_predictions):
        if item["label"] != "patient_name":
            continue
        
        text = item["text"].strip()
        confidence = float(item["confidence"])
        
        if not valid_patient_name(text):
            continue
        
        score = patient_name_score(text, confidence)
        
        # Boost score if near a label indicator
        if label_positions:
            min_distance = min(abs(idx - label_idx) for label_idx in label_positions)
            if min_distance <= 2:  # Within 2 tokens of label
                score += 0.3
            elif min_distance <= 5:  # Within 5 tokens
                score += 0.15
        
        candidates.append((text, score))
    
    # STEP 3: Look for text with "Patient:" or "Patient Name:" prefix in ANY label (mislabeled cases)
    for idx, item in enumerate(ml_predictions):
        text = item["text"].strip()
        confidence = float(item["confidence"])
        
        # Check if text starts with "Patient:" or "Patient Name:" and extract the actual name
        # Use ^ to ensure it starts with "Patient" (not just contains it)
        patient_match = re.match(r'^patient\s*(?:name)?:?\s*(.+)', text, re.IGNORECASE)
        if patient_match:
            extracted_name = patient_match.group(1).strip()
            # Basic validation: should have alphabetic characters and reasonable length
            # AND should not be a header/label itself
            if (extracted_name and 
                len(extracted_name) >= 3 and 
                sum(c.isalpha() for c in extracted_name) >= len(extracted_name) * 0.5 and
                valid_patient_name(extracted_name)):  # Validate the extracted name
                score = patient_name_score(extracted_name, confidence * 0.9)
                # Boost score since this is a clear label indicator
                score += 0.3
                candidates.append((extracted_name, score))
    
    # STEP 4: If still no candidates, look for text immediately after label positions
    if not candidates and label_positions:
        for label_idx in label_positions:
            # Check next 1-3 tokens after label
            for offset in range(1, 4):
                check_idx = label_idx + offset
                if check_idx < len(ml_predictions):
                    item = ml_predictions[check_idx]
                    text = item["text"].strip()
                    confidence = float(item["confidence"])
                    
                    if valid_patient_name(text):
                        # Higher boost for closer proximity
                        proximity_boost = 0.4 if offset == 1 else (0.25 if offset == 2 else 0.15)
                        score = patient_name_score(text, confidence * 0.7) + proximity_boost
                        candidates.append((text, score))
    
    if not candidates:
        return {
            "patient_name": None,
            "confidence": None,
            "status": "rejected",
            "reason": "no_valid_patient_name_candidate"
        }
    
    best_text, best_score = max(candidates, key=lambda x: x[1])
    
    # Normalize score using sigmoid
    normalized_score = sigmoid_normalize(best_score)
    
    if best_score >= CONF_HIGH:
        return {
            "patient_name": best_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted",
            "reason": "high_confidence"
        }
    
    if CONF_LOW <= best_score < CONF_HIGH:
        return {
            "patient_name": best_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted_low_confidence",
            "reason": "low_confidence_but_valid"
        }
    
    return {
        "patient_name": None,
        "confidence": None,
        "status": "rejected",
        "reason": "confidence_too_low"
    }


# ================= DATE VALIDATION =================

def valid_date(text: str) -> bool:
    """Validate date format"""
    t = text.strip()
    
    # Common date patterns
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY or DD-MM-YYYY
        r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD
        r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}',  # DD Month YYYY
        r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}',  # Month DD, YYYY
        r'\d{1,2}[-][A-Za-z]{3}[-]\d{2,4}',  # DD-Mon-YY or DD-Mon-YYYY (23-Oct-18)
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, t):
            return True
    
    return False


def date_score(text: str, confidence: float) -> float:
    """Score date"""
    score = confidence
    
    # Strong date pattern match
    if valid_date(text):
        score += 0.2
    
    # Contains month names
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    if any(month in text.lower() for month in months):
        score += 0.15
    
    return min(score, 1.0)


def validate_date(ml_predictions: List[Dict]) -> Dict:
    """Validate date from ML predictions"""
    candidates = []
    
    for item in ml_predictions:
        if item["label"] != "date":
            continue
        
        text = item["text"].strip()
        confidence = float(item["confidence"])
        
        if not valid_date(text):
            continue
        
        score = date_score(text, confidence)
        candidates.append((text, score))
    
    # If no date candidates found, look for date patterns in other labels with lower threshold
    if not candidates:
        for item in ml_predictions:
            text = item["text"].strip()
            confidence = float(item["confidence"])
            
            # Check if text contains a date pattern even if labeled differently
            if valid_date(text) and confidence >= 0.3:
                score = date_score(text, confidence)
                candidates.append((text, score * 0.8))  # Reduce score since it wasn't labeled as date
    
    if not candidates:
        return {
            "date": None,
            "confidence": None,
            "status": "rejected",
            "reason": "no_valid_date_candidate"
        }
    
    best_text, best_score = max(candidates, key=lambda x: x[1])
    
    # Extract just the date portion from the text
    date_text = best_text
    
    # First, try to extract just the date pattern
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY or DD-MM-YYYY
        r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD
        r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}',  # DD Month YYYY
        r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}',  # Month DD, YYYY
        r'\d{1,2}[-][A-Za-z]{3}[-]\d{2,4}',  # DD-Mon-YY or DD-Mon-YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_text)
        if match:
            date_text = match.group(0)
            break
    
    # Then clean any remaining prefixes
    date_prefixes = [r'bill\s+date:\s*', r'discharge\s+date:\s*', r'admission\s+date:\s*', 
                     r'refill\s+due:\s*', r'due:\s*',
                     r'date:\s*', r'on\s+', r'discahrge\s+dt\.?/time\s*:\s*',
                     r'discharge\s+dt\.?/time\s*:\s*', r'dt\.?/time\s*:\s*']
    for prefix in date_prefixes:
        date_text = re.sub(prefix, '', date_text, flags=re.IGNORECASE).strip()
    
    # Normalize score using sigmoid
    normalized_score = sigmoid_normalize(best_score)
    
    if best_score >= CONF_HIGH:
        return {
            "date": date_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted",
            "reason": "high_confidence"
        }
    
    if CONF_LOW <= best_score < CONF_HIGH:
        return {
            "date": date_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted_low_confidence",
            "reason": "low_confidence_but_valid"
        }
    
    return {
        "date": None,
        "confidence": None,
        "status": "rejected",
        "reason": "confidence_too_low"
    }


# ================= ADDRESS VALIDATION =================

def valid_address(text: str) -> bool:
    """Validate address structure - must have actual address indicators"""
    t = text.strip()
    tl = t.lower()
    
    if len(t) < 5:
        return False
    
    # Reject labels/headers with colons (like "Payment Mode:", "Status:")
    if ':' in text and len(text) < 50:
        return False
    
    # Reject GSTIN and tax IDs
    if re.search(r'gstin|gst\s*no|pan\s*no|tin\s*no|^\d{2}[a-z]{5}\d{4}[a-z]\d[a-z]\d', tl):
        return False
    
    # Reject service/charge descriptions, account/statement headers, medicine names, and payment info
    service_patterns = [r'charges?', r'test', r'panel', r'service', r'fee', 
                       r'consultation', r'lab', r'diagnostic', r'payment\s*status',
                       r'paid', r'unpaid', r'pending', r'status:', r'taxable',
                       r'\bamount\b', r'\btax\b', r'\bgst\b', r'cgst', r'sgst', r'igst',
                       r'account', r'statement',
                       r'\d+\s*mg', r'\d+\s*ml', r'\d+\s*gm',
                       r'strip\s+of', r'\(strip', r'tablet', r'capsule', r'syrup',
                       r'amoxicillin', r'paracetamol', r'aspirin', r'ibuprofen',
                       r'\bmedicine\b', r'\bdrug\b',
                       r'payment\s*mode', r'payment\s*method', r'\bupi\b', r'credit', r'debit',
                       r'cash', r'card', r'paytm', r'gpay', r'phonepe', r'net\s*banking']
    for pattern in service_patterns:
        if re.search(pattern, tl):
            return False
    
    # Reject if looks like bed/room/ward info or phone/call instructions
    non_address_patterns = [r'bed\s*no', r'room\s*no', r'ward', r'icu\d+',
                           r'timing:', r'ph:', r'mob\.?\s*no', r'\d{2}:\d{2}\s*[ap]m',
                           r'closed:', r'open:', r'call\s+us', r'phone\s+number',
                           r'contact\s+us', r'if\s+you', r'would\s+like']
    for pattern in non_address_patterns:
        if re.search(pattern, tl):
            return False
    
    # MUST contain address keywords or pincode - this is now required, not optional
    if looks_like_address(t):
        return True
    
    # If no address keywords found, reject it
    return False


def address_score(text: str, confidence: float) -> float:
    """Score address - heavily favor longer text"""
    score = confidence
    
    # Contains address keywords
    if looks_like_address(text):
        score += 0.2
    
    # Contains pincode
    if re.search(r'\b\d{6}\b', text):
        score += 0.15
    
    # Address is typically the longest sentence - heavily boost longer text
    text_length = len(text)
    if text_length > 50:
        score += 0.3
    elif text_length > 30:
        score += 0.2
    elif text_length > 20:
        score += 0.1
    
    # Word count bonus
    word_count = len(text.split())
    if word_count >= 5:
        score += 0.15
    elif word_count >= 3:
        score += 0.1
    
    return min(score, 1.0)


def validate_address(ml_predictions: List[Dict]) -> Dict:
    """Validate address from ML predictions"""
    candidates = []
    
    # First pass: look for address-labeled items
    for item in ml_predictions:
        if item["label"] != "address":
            continue
        
        text = item["text"].strip()
        confidence = float(item["confidence"])
        
        if not valid_address(text):
            continue
        
        score = address_score(text, confidence)
        candidates.append((text, score))
    
    # Second pass: if no good address found, look for text with actual address keywords only
    if not candidates:
        all_texts = []
        for item in ml_predictions:
            text = item["text"].strip()
            confidence = float(item["confidence"])
            
            # Skip if too short or invalid
            if len(text) < 15:
                continue
            
            # MUST have address keywords to be considered
            if not looks_like_address(text):
                continue
            
            # Skip GSTIN/tax patterns
            if re.search(r'gstin|gst\s*no|pan\s*no|tin\s*no', text.lower()):
                continue
            
            # Skip service/charge descriptions and payment information
            if re.search(r'charges?|test|panel|service|fee|consultation|lab|diagnostic|payment\s*status|paid|unpaid|pending|status:|taxable|\bamount\b|\btax\b|\bgst\b|cgst|sgst|igst|account|statement|payment\s*mode|payment\s*method|\bupi\b|credit|debit|cash|card|paytm|gpay|phonepe|net\s*banking', text.lower()):
                continue
            
            # Check for non-address patterns (including doctor names)
            t_lower = text.lower()
            if any(pattern in t_lower for pattern in ['bed no', 'room no', 'ward', 'icu', 'timing:', 'ph:', 'dr.', 'dr ', 'doctor', 'call us', 'phone number', 'contact us']):
                continue
            
            # Reject medical professional titles in parentheses
            if re.search(r'\((neurologist|cardiologist|surgeon|physician|practitioner|consultant|specialist)', t_lower):
                continue
                continue
            
            # Skip if it's already captured by other fields
            if contains_hospital_keyword(text):
                continue
            
            # CRITICAL: Reject if it looks like a medicine/product (even if has address-like patterns)
            medicine_patterns = [r'\d+\s*mg', r'\d+\s*ml', r'\d+\s*gm',
                               r'strip\s+of', r'\(strip', r'tablet', r'capsule', r'syrup',
                               r'amoxicillin', r'paracetamol', r'aspirin', r'ibuprofen',
                               r'\bmedicine\b', r'\bdrug\b', r'\bmedication\b']
            if any(re.search(pattern, text.lower()) for pattern in medicine_patterns):
                continue
            
            # Skip generic headers
            if looks_like_generic_header(text) or looks_like_table_header(text):
                continue
            
            # At this point, it has address keywords and passes all filters
            text_length = len(text)
            word_count = len(text.split())
            
            # Boost score based on length
            length_score = min(text_length / 100.0, 0.4)
            score = confidence * 0.8 + length_score + 0.3  # Higher base score since it has address keywords
            
            all_texts.append((text, score, text_length))
        
        # Pick the longest one if available
        if all_texts:
            # Sort by length descending, then by score
            all_texts.sort(key=lambda x: (x[2], x[1]), reverse=True)
            best_text, best_score, _ = all_texts[0]
            candidates.append((best_text, best_score))
    
    if not candidates:
        return {
            "address": None,
            "confidence": None,
            "status": "rejected",
            "reason": "no_valid_address_candidate"
        }
    
    best_text, best_score = max(candidates, key=lambda x: x[1])
    
    # Clean the address text by removing common prefixes
    address_text = best_text
    address_prefixes = [r'address:\s*', r'location:\s*', r'addr:\s*']
    for prefix in address_prefixes:
        address_text = re.sub(prefix, '', address_text, flags=re.IGNORECASE).strip()
    
    # Normalize score using sigmoid
    normalized_score = sigmoid_normalize(best_score)
    
    if best_score >= CONF_HIGH:
        return {
            "address": address_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted",
            "reason": "high_confidence"
        }
    
    if CONF_LOW <= best_score < CONF_HIGH:
        return {
            "address": address_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted_low_confidence",
            "reason": "low_confidence_but_valid"
        }
    
    return {
        "address": None,
        "confidence": None,
        "status": "rejected",
        "reason": "confidence_too_low"
    }


# ================= TOTAL AMOUNT VALIDATION =================

def valid_total_amount(text: str) -> bool:
    """Validate total amount structure"""
    t = text.strip()
    
    # Should contain numbers
    if not any(c.isdigit() for c in t):
        return False
    
    # Reject if it looks like a phone number (10 consecutive digits, no decimal)
    if re.match(r'^\d{10}$', t):
        return False
    
    # Reject year-like numbers (2020-2030 range without decimals)
    if re.match(r'^(202[0-9]|203[0])$', t):
        return False
    
    # Extract numeric value to check if it's valid
    numeric_text = re.sub(r'[^0-9.]', '', t)
    try:
        amount_value = float(numeric_text) if numeric_text else 0
    except:
        amount_value = 0
    
    # Reject zero or very small amounts (medical bills are never 0 or less than 1)
    if amount_value < 1.0:
        return False
    
    # Reject unrealistically large amounts (likely phone numbers or IDs)
    # Medical bills rarely exceed 10 million (1 crore)
    if amount_value > 10000000:
        return False
    
    # Reject invoice-like numbers: 5+ digits without decimal (e.g., 98765, 12345)
    # But allow 3-4 digit amounts (like 509, 1234) as they could be valid totals
    if re.match(r'^[₹$€£]?\s*\d{5,}$', t):  # 5+ digits, no decimal
        # However, if it's near "total" context, allow it
        return False
    
    # Reject single digit amounts (likely not a total)
    if re.match(r'^\d$', t):
        return False
    
    # Common amount patterns
    amount_patterns = [
        r'^\d{2,}\.?\d*$',  # At least 2 digits
        r'^[\$₹€£]\s*\d+\.?\d*$',  # With currency symbol
        r'^\d{1,3}(,\d{3})+\.?\d*$',  # With comma separators
        r'^\d+\.\d{2}$',  # With decimal (like 50.00)
    ]
    
    for pattern in amount_patterns:
        if re.search(pattern, t):
            return True
    
    return False


def total_amount_score(text: str, confidence: float, context: str = "") -> float:
    """Score total amount - prefer amounts with decimals and near 'total' keywords"""
    score = confidence
    
    # Extract numeric value for comparison
    numeric_text = re.sub(r'[^0-9.]', '', text)
    try:
        amount_value = float(numeric_text) if numeric_text else 0
    except:
        amount_value = 0
    
    # STRONGLY boost if near "total" keywords (these are almost always the final total)
    context_lower = context.lower()
    if re.search(r'\b(grand|net|final)\s*total\b', context_lower):
        score += 0.9  # Huge boost for "Grand Total", "Net Total", "Final Total"
    elif re.search(r'\btotal\b', context_lower):
        score += 0.7  # Strong boost for "Total"
    elif re.search(r'\b(payable|due|balance)\b', context_lower):
        score += 0.6  # Boost for "Amount Payable", "Amount Due"
    
    # Boost amounts with decimal points (real money amounts)
    # But don't penalize amounts without decimals (like 509, 1234)
    if '.' in text and re.search(r'\d+\.\d{2}', text):
        score += 0.3  # Moderate preference for amounts like 532.00, 1234.50
    
    # Heavily penalize zero or near-zero amounts (they can't be valid totals)
    if amount_value < 1.0:
        score -= 1.0  # Massive penalty
    
    # Penalize round thousands (like 10,000 or 5,000) as they're often subtotals
    if amount_value > 0 and amount_value % 10000 == 0:
        score -= 0.2  # Penalize round amounts
    
    # For very large amounts (>10k), give bonuses as they're likely grand totals
    # But for smaller amounts (<10k), don't penalize - could be valid pharmacy bills
    if amount_value > 100000:
        score += 0.3  # Very large amounts (100k+)
    elif amount_value > 50000:
        score += 0.25
    elif amount_value > 15000:
        score += 0.15
    
    # Contains currency symbols
    if any(sym in text for sym in ['₹', '$', '€', '£']):
        score += 0.15
    
    # Has decimal point (more specific)
    if '.' in text:
        score += 0.1
    
    # Has comma separators but is NOT a round thousand
    if ',' in text and amount_value % 10000 != 0:
        score += 0.15
    
    return min(score, 1.0)


def validate_total_amount(ml_predictions: List[Dict]) -> Dict:
    """Validate total amount - uses proximity to total/grand total labels"""
    candidates = []
    
    # STEP 1: Find label positions for "Grand Total:", "Total Amount:", "Net Total:", etc.
    label_positions = []
    strong_label_positions = []  # Grand Total, Net Total, Final Total
    
    for idx, item in enumerate(ml_predictions):
        text = item["text"].strip().lower()
        
        # Strong total indicators (flexible matching for OCR errors like "T0TAL" instead of "TOTAL")
        # Match "TOTAL", "T0TAL", "TDTAL", etc.
        if re.search(r'\b(grand|net|final)\s*t[o0]tal\b', text):
            strong_label_positions.append(idx)
            label_positions.append(idx)
        # Regular total indicators (but NOT sub-total - that's not the final total)
        # Also handle OCR errors like "T0TAL"
        elif re.search(r'\bt[o0]tal\s*(amount|payable|due)?\b', text) and 'sub' not in text:
            label_positions.append(idx)
        # Amount due/payable
        elif re.search(r'\b(amount|payment)\s*(due|payable)\b', text):
            label_positions.append(idx)
    
    # STEP 2: Search ALL predictions for amounts, prioritizing those near labels
    for idx, item in enumerate(ml_predictions):
        text = item["text"].strip()
        confidence = float(item["confidence"])
        
        # Try to extract amount from this text
        # Use negative lookbehind/lookahead to avoid matching digits inside words (like "T0TAL")
        amount_match = re.search(r'(?<![a-zA-Z0-9])[\u20b9$\u20ac\u00a3]?\s*([0-9,]+\.?[0-9]*)(?![a-zA-Z])', text)
        if amount_match:
            amount_text = amount_match.group(1)
            
            if valid_total_amount(amount_text):
                # Get context for scoring
                context_items = ml_predictions[max(0, idx-2):min(len(ml_predictions), idx+3)]
                context = " ".join([c["text"] for c in context_items])
                
                score = total_amount_score(amount_text, confidence, context)
                
                # Extract numeric value
                numeric_text = re.sub(r'[^0-9.]', '', amount_text)
                try:
                    amount_value = float(numeric_text) if numeric_text else 0
                except:
                    amount_value = 0
                
                if amount_value < 1:
                    continue  # Skip zero amounts
                
                # CRITICAL: Add proximity boost for amounts near total labels
                proximity_boost = 0
                if strong_label_positions:
                    min_distance = min(abs(idx - label_idx) for label_idx in strong_label_positions)
                    if min_distance == 0:  # Same token as label (e.g., "Grand Total: 1071")
                        proximity_boost = 2.5  # Massive boost
                    elif min_distance == 1:  # Immediately after label
                        proximity_boost = 2.0
                    elif min_distance == 2:
                        proximity_boost = 1.5
                    elif min_distance <= 3:
                        proximity_boost = 1.0
                
                elif label_positions:
                    min_distance = min(abs(idx - label_idx) for label_idx in label_positions)
                    if min_distance == 0:
                        proximity_boost = 1.8
                    elif min_distance == 1:
                        proximity_boost = 1.5
                    elif min_distance == 2:
                        proximity_boost = 1.0
                    elif min_distance <= 3:
                        proximity_boost = 0.6
                
                final_score = score + proximity_boost
                candidates.append((amount_text, final_score, amount_value))

    
    # STEP 3: If multiple candidates, prefer largest amount as tiebreaker
    if len(candidates) > 1:
        max_amount = max(c[2] for c in candidates)
        
        updated_candidates = []
        for text, score, amount_value in candidates:
            # Boost the largest amount
            if amount_value == max_amount:
                score += 0.5
            elif amount_value < max_amount * 0.1:  # Less than 10% of max
                score -= 1.0  # Heavy penalty for tiny amounts
            
            updated_candidates.append((text, score, amount_value))
        
        candidates = updated_candidates
    
    if not candidates:
        return {
            "total_amount": None,
            "confidence": None,
            "status": "rejected",
            "reason": "no_valid_total_amount_candidate"
        }
    
    # Pick the best (highest score, which now heavily favors largest amounts)
    best_text, best_score, best_amount = max(candidates, key=lambda x: x[1])
    
    # Clean up OCR errors: "7" at start is often misread rupee symbol (₹)
    cleaned_text = best_text
    # Pattern: 7 followed by digits with optional decimal (like 7650.00, 7180.00, 7509)
    # This is likely ₹650.00, ₹180.00, ₹509 misread
    if cleaned_text.startswith('7') and len(cleaned_text) >= 4:
        # Remove the leading "7" if it's followed by digits
        if cleaned_text[1].isdigit():
            cleaned_text = cleaned_text[1:]  # Remove leading "7"
    
    # Format amount with commas and 2 decimal places
    numeric_text = re.sub(r'[^0-9.]', '', cleaned_text)
    try:
        amount_float = float(numeric_text) if numeric_text else 0
        # Format with commas and exactly 2 decimal places
        cleaned_text = f"{amount_float:,.2f}"
    except:
        pass  # Keep original if parsing fails
    
    # Normalize score using sigmoid
    normalized_score = sigmoid_normalize(best_score)
    
    if best_score >= CONF_HIGH:
        return {
            "total_amount": cleaned_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted",
            "reason": "high_confidence"
        }
    
    if CONF_LOW <= best_score < CONF_HIGH:
        return {
            "total_amount": cleaned_text,
            "confidence": round(normalized_score, 2),
            "status": "accepted_low_confidence",
            "reason": "low_confidence_but_valid"
        }
    
    return {
        "total_amount": None,
        "confidence": None,
        "status": "rejected",
        "reason": "confidence_too_low"
    }
