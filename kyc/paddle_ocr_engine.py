ocr = None

def get_ocr_engine():
    """Lazy load PaddleOCR to avoid slow imports"""
    global ocr
    if ocr is None:
        import paddle
        from paddleocr import PaddleOCR
        
        paddle.set_device("cpu")
        print("🚀 Loading PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        print("✨ PaddleOCR Loaded Successfully!")
    return ocr


def extract_text(image_path: str):
    """
    Runs OCR and returns full clean text only (single string)
    """
    ocr_engine = get_ocr_engine()
    result = ocr_engine.ocr(image_path)

    rec_texts = []

    # Handle new PaddleOCR format (dictionary style)
    if result and isinstance(result[0], dict):
        rec_texts = result[0].get("rec_texts", [])

    return "\n".join(rec_texts)


def extract_for_layoutlm(image_path: str):
    """
    Convert PaddleOCR output into LayoutLMv3 format with words and bounding boxes
    Handles both old (list) and new (dict) PaddleOCR formats
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]

    ocr_engine = get_ocr_engine()
    result = ocr_engine.ocr(image_path)

    words = []
    bboxes = []

    if not result or not result[0]:
        return {"words": [], "bboxes": [], "full_text": "", "image_size": {"width": w, "height": h}}

    # Handle PaddleOCR format - check if result has dict-like attributes
    try:
        # Try to access as dict or dict-like object (OCRResult)
        rec_texts = result[0].get("rec_texts", []) if hasattr(result[0], 'get') else getattr(result[0], 'rec_texts', [])
        rec_boxes = result[0].get("rec_boxes", []) if hasattr(result[0], 'get') else getattr(result[0], 'rec_boxes', [])
        
        print(f"DEBUG: rec_texts length = {len(rec_texts) if rec_texts is not None else 'None'}")
        print(f"DEBUG: rec_boxes length = {len(rec_boxes) if rec_boxes is not None else 'None'}")
        if rec_texts is not None and len(rec_texts) > 0:
            print(f"DEBUG: First text = {rec_texts[0]}")
            print(f"DEBUG: First box = {rec_boxes[0]}")
        
        if len(rec_texts) > 0 and len(rec_boxes) > 0:
            for text, box in zip(rec_texts, rec_boxes):
                try:
                    # box is already a numpy array with 4 corner points
                    x1, y1 = box[0]
                    x3, y3 = box[2]

                    bbox = [
                        int((x1 / w) * 1000),
                        int((y1 / h) * 1000),
                        int((x3 / w) * 1000),
                        int((y3 / h) * 1000)
                    ]

                    words.append(text)
                    bboxes.append(bbox)
                except Exception as e:
                    print(f"DEBUG: Error processing box: {e}, box shape: {box.shape if hasattr(box, 'shape') else 'no shape'}")
                    continue
        # If no data extracted, try old format
        elif isinstance(result[0], list):
            for line in result:
                try:
                    box = line[0]      # list of 4 points
                    text = line[1][0]  # text label

                    x1, y1 = box[0]
                    x3, y3 = box[2]

                    bbox = [
                        int((x1 / w) * 1000),
                        int((y1 / h) * 1000),
                        int((x3 / w) * 1000),
                        int((y3 / h) * 1000)
                    ]

                    words.append(text)
                    bboxes.append(bbox)
                except:
                    continue
    except Exception as e:
        print(f"Warning: Could not extract OCR data: {e}")

    full_text = "\n".join(words)

    return {
        "words": words,
        "bboxes": bboxes,
        "full_text": full_text,
        "image_size": {"width": w, "height": h}
    }
