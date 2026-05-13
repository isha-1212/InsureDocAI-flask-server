import torch
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader, total=len(data_loader)):
        # Move batch to device
        for k, v in batch.items():
            batch[k] = v.to(device)

        optimizer.zero_grad()

        logits, loss = model(**batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            logits, loss = model(**batch)

            total_loss += loss.item()

    return total_loss / len(data_loader)
