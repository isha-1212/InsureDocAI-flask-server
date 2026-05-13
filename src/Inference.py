# inference.py

import os
import sys
import torch
import re
from PIL import Image
import torch.nn.functional as F
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3Processor
)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from trainer import ModelModule
from utils import dataSetFormat

_PROCESSOR = None
_MODEL = None


# ================= CONFIG =================

MODEL_PATH = os.environ.get(
    "ML_MODEL_PATH",
    os.path.join(_SRC_DIR, "SplittedTrainedMODEL.bin")
)
TOKENIZER_PATH = "microsoft/layoutlmv3-base"

LABEL_MAP = {
    0: "ignored",
    1: "hospital_name",
    2: "patient_name",
    3: "date",
    4: "address",
    5: "total_amount"
}

CONF_THRESHOLD = 0.6

# =========================================


def _get_inference_components():
    global _PROCESSOR, _MODEL

    if _PROCESSOR is None:
        _PROCESSOR = LayoutLMv3Processor(
            tokenizer=LayoutLMv3TokenizerFast.from_pretrained(TOKENIZER_PATH),
            feature_extractor=LayoutLMv3FeatureExtractor(apply_ocr=False)
        )

    if _MODEL is None:
        _MODEL = ModelModule(num_labels=6)
        _MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _MODEL.eval()

    return _PROCESSOR, _MODEL


def run_inference(image_path):

    # ---------- Reuse loaded components ----------
    processor, model = _get_inference_components()

    # ---------- Load image ----------
    image = Image.open(image_path).convert("RGB")

    # ---------- OCR ----------
    test_dict, width, height = dataSetFormat(image)

    # ---------- Encode ----------
    encoding = processor(
        image,
        test_dict["tokens"],
        boxes=test_dict["bboxes"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    # ---------- Inference ----------
    with torch.no_grad():
        logits, _ = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            bbox=encoding["bbox"],
            pixel_values=encoding["pixel_values"]
        )

    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1).squeeze()
    conf = torch.max(probs, dim=-1).values.squeeze()

    # ---------- Remove subword tokens ----------
    offset = encoding["offset_mapping"].squeeze()
    valid = offset[:, 0] == 0

    tokens = test_dict["tokens"]
    final = []

    token_idx = 0
    for i in range(len(valid)):
        if valid[i] and token_idx < len(tokens):
            final.append({
                "text": tokens[token_idx],
                "label": LABEL_MAP[int(preds[i])],
                "confidence": float(conf[i])
            })
            token_idx += 1

    # ❌ NO PRINT HERE
    return {
        "ml_predictions": final,
        "ocr_tokens": test_dict["tokens"]
    }


if __name__ == "__main__":
    import sys
    import glob
    import os
    
    # Check if image path is provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if os.path.isfile(image_path):
            print(f"\n{'='*50}")
            print(f"Running inference on: {image_path}")
            print(f"{'='*50}\n")
            
            result = run_inference(image_path)
            
            print("\n=== OCR Results ===")
            print(f"Total tokens detected: {len(result['ocr_tokens'])}")
            
            print("\n=== ML Predictions ===")
            for item in result["ml_predictions"]:
                if item["label"] != "ignored" and item["confidence"] > CONF_THRESHOLD:
                    print(f"{item['label']:20s} | {item['text']:30s} | conf: {item['confidence']:.3f}")
        else:
            print(f"Error: File not found: {image_path}")
    else:
        # Default: look for images in data/hospital/images or data/pharmacy/images
        image_patterns = [
            r"C:\vscode\CVprojects\SGP6\data\hospital\images\*.jpg",
            r"C:\vscode\CVprojects\SGP6\data\hospital\images\*.png",
            r"C:\vscode\CVprojects\SGP6\data\pharmacy\images\*.jpg",
            r"C:\vscode\CVprojects\SGP6\data\pharmacy\images\*.png"
        ]
        
        images = []
        for pattern in image_patterns:
            images.extend(glob.glob(pattern))
        
        if not images:
            print("\nNo images found in data/hospital/images/ or data/pharmacy/images/")
            print("\nUsage:")
            print("  python Inference.py <path_to_image>")
            print("\nExample:")
            print("  python Inference.py C:\\vscode\\CVprojects\\SGP6\\data\\hospital\\images\\sample.jpg")
        else:
            print(f"\nFound {len(images)} image(s). Running inference on the first one...\n")
            result = run_inference(images[0])
            
            print(f"\n{'='*50}")
            print(f"Image: {os.path.basename(images[0])}")
            print(f"{'='*50}\n")
            
            print("=== OCR Results ===")
            print(f"Total tokens detected: {len(result['ocr_tokens'])}")
            
            print("\n=== ML Predictions ===")
            for item in result["ml_predictions"]:
                if item["label"] != "ignored" and item["confidence"] > CONF_THRESHOLD:
                    print(f"{item['label']:20s} | {item['text']:30s} | conf: {item['confidence']:.3f}")
