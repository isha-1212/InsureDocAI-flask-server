import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    LayoutLMv3ImageProcessor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3Processor
)

from loader import LayoutLMv3Dataset
from trainer import ModelModule
from engine import train_fn, eval_fn


# ================= CONFIG =================

TRAIN_JSON = r"C:\vscode\CVprojects\SGP6\inputs\Training_layoutLMV3.json"
MODEL_BASE = "microsoft/layoutlmv3-base"
SAVE_DIR = r"C:\vscode\CVprojects\SGP6\src"

BATCH_SIZE = 2
EPOCHS = 30
LR = 5e-5

NUM_LABELS = 6   # ignored + 5 entity labels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================


def main():
    # ---------- Processor ----------
    image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        MODEL_BASE,
        ignore_mismatched_sizes=True
    )
    processor = LayoutLMv3Processor(
        image_processor=image_processor,
        tokenizer=tokenizer
    )

    # ---------- Dataset & Loader ----------
    train_dataset = LayoutLMv3Dataset(
        json_path=TRAIN_JSON,
        processor=processor,
        max_length=512
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ---------- Model ----------
    model = ModelModule(num_labels=NUM_LABELS, model_name_or_path=MODEL_BASE)
    model.to(DEVICE)

    # ---------- Optimizer ----------
    optimizer = AdamW(model.parameters(), lr=LR)

    best_loss = float("inf")
    loss_history = []

    # ---------- Training Loop ----------
    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch + 1}/{EPOCHS}")

        train_loss = train_fn(
            train_loader,
            model,
            optimizer,
            DEVICE
        )

        print(f"Train loss: {train_loss:.4f}")
        loss_history.append(train_loss)

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                model.state_dict(),
                f"{SAVE_DIR}/best_model.bin"
            )

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                f"{SAVE_DIR}/model_epoch_{epoch + 1}.bin"
            )

        # Optional evaluation on same data (small dataset)
        eval_loss = eval_fn(
            train_loader,
            model,
            DEVICE
        )
        print(f"Eval loss: {eval_loss:.4f}")

    # ---------- Save loss curve ----------
    np.save(f"{SAVE_DIR}/loss_history.npy", np.array(loss_history))
    print("\n✅ Training completed successfully")


if __name__ == "__main__":
    main()
