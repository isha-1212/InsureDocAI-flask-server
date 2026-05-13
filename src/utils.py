import json
import os

# Paddle reads CPU backend flags during import. Force conservative settings
# to avoid the oneDNN fused-conv path that is crashing on this Windows setup.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from paddleocr import PaddleOCR
import pytesseract


# ================= OCR =================
# New PaddleOCR API (NO rec, NO use_angle_cls)

ocr = PaddleOCR(lang="en", enable_mkldnn=False, cpu_threads=1)


def _normalize_box(x1, y1, x2, y2, width, height):
    x1_norm = int((x1 / width) * 1000)
    y1_norm = int((y1 / height) * 1000)
    x2_norm = int((x2 / width) * 1000)
    y2_norm = int((y2 / height) * 1000)

    return [
        max(0, min(x1_norm, 1000)),
        max(0, min(y1_norm, 1000)),
        max(0, min(x2_norm, 1000)),
        max(0, min(y2_norm, 1000)),
    ]


def _extract_with_tesseract(img, width, height):
    data = pytesseract.image_to_data(
        img,
        lang="eng",
        output_type=pytesseract.Output.DICT
    )

    tokens = []
    bboxes = []
    for i, text in enumerate(data.get("text", [])):
        token = (text or "").strip()
        if not token:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1

        if conf < 0:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        if w <= 0 or h <= 0:
            continue

        tokens.append(token)
        bboxes.append(_normalize_box(x, y, x + w, y + h, width, height))

    return {
        "tokens": tokens,
        "bboxes": bboxes,
        "img_path": img
    }


# ================= JSON READER =================

def read_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================= TRAIN DATA FORMAT =================
# Converts Training_layoutLMV3.json → internal format

def train_data_format(json_list: list):
    final_list = []

    for idx, item in enumerate(json_list):
        data = {
            "id": idx,
            "img_path": item["file_name"],
            "tokens": [],
            "bboxes": [],
            "ner_tag": []
        }

        for ann in item["annotations"]:
            data["tokens"].append(ann["text"])
            data["bboxes"].append(ann["box"])       # already [x1,y1,x2,y2]
            data["ner_tag"].append(ann["label_id"]) # ✅ INTEGER labels

        final_list.append(data)

    return final_list


# ================= OCR → INFERENCE FORMAT =================

def dataSetFormat(img):
    """
    Runs OCR on a PIL image and returns:
    tokens, bboxes (normalized to 0-1000), image dimensions
    """

    width, height = img.size
    test_dict = {
        "tokens": [],
        "bboxes": [],
        "img_path": img
    }

    try:
        ocr_result = ocr.ocr(np.asarray(img))

        for res in ocr_result:
            polys = res["dt_polys"]
            texts = res["rec_texts"]

            for poly, text in zip(polys, texts):
                x_coords = [p[0] for p in poly]
                y_coords = [p[1] for p in poly]

                x1 = min(x_coords)
                y1 = min(y_coords)
                x2 = max(x_coords)
                y2 = max(y_coords)

                test_dict["tokens"].append(text)
                test_dict["bboxes"].append(_normalize_box(x1, y1, x2, y2, width, height))
    except Exception:
        test_dict = _extract_with_tesseract(img, width, height)

    return test_dict, width, height


# ================= VISUALIZATION =================

def plot_img(img, bbox_list, label_list, prob_list, width, height):
    plt.imshow(img)
    ax = plt.gca()

    for i, box in enumerate(bbox_list):
        x1, y1, x2, y2 = box.tolist()

        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            f"{label_list[i]} ({prob_list[i]:.2f})",
            color="black",
            bbox=dict(facecolor="white", alpha=0.6),
            fontsize=8
        )

    plt.axis("off")
    plt.show()
