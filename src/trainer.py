import torch
import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification


class ModelModule(nn.Module):
    def __init__(self, num_labels: int, model_name_or_path: str = "microsoft/layoutlmv3-base"):
        super().__init__()

        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        bbox,
        pixel_values,
        labels=None
    ):
        """
        Returns:
        - logits: (batch, seq_len, num_labels)
        - loss: scalar
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )

        return outputs.logits, outputs.loss
