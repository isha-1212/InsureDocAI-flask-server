import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class LayoutLMv3Dataset(Dataset):
    def __init__(self, json_path, processor, max_length=512):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]

            # -------- Image --------
            image = Image.open(item["file_name"]).convert("RGB")

            # -------- Tokens, boxes, labels --------
            words = [str(ann["text"]).strip() for ann in item["annotations"] if ann["text"] and str(ann["text"]).strip()]
            boxes = [ann["box"] for ann in item["annotations"] if ann["text"] and str(ann["text"]).strip()]
            labels = [ann["label_id"] for ann in item["annotations"] if ann["text"] and str(ann["text"]).strip()]
            
            # Skip if no valid annotations
            if not words:
                print(f"Warning: Document {idx} has no valid annotations, skipping")
                # Return a dummy entry with a single token
                words = [" "]
                boxes = [[0, 0, 1, 1]]
                labels = [0]

            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                word_labels=labels,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "bbox": encoding["bbox"].squeeze(0),
                "pixel_values": encoding["pixel_values"].squeeze(0),
                "labels": encoding["labels"].squeeze(0)
            }
        except Exception as e:
            print(f"Error processing document {idx}: {e}")
            print(f"File: {item.get('file_name', 'unknown')}")
            print(f"Number of annotations: {len(item.get('annotations', []))}")
            raise
