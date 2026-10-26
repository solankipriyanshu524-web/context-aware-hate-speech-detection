"""
DEVIPUTRA AI // PURE PYTORCH & TRANSFORMERS ENGINE
Fine-tunes Multilingual HingHate Transformer on Code-Mixed Hinglish Slangs & Hate Speech.
Features:
- Pure Python & PyTorch (zero external DLL blocks from pandas/sklearn)
- Supports PyTorch CUDA / Tensor Cores / CPU execution
- Stratified 85/15 Train-Val split with F1-Score & Accuracy tracking
- Saves production-ready checkpoint into models/hing_hate_bert_rtx5050/
"""

import os
import sys
import csv
import random
from typing import Dict, List, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch

LABEL_TO_ID = {"Normal": 0, "Offensive": 1, "Hate Speech": 2}
ID_TO_LABEL = {0: "Normal", 1: "Offensive", 2: "Hate Speech"}

class HinglishTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data_clean(filepath: str) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                txt = row[0].strip()
                lbl = row[1].strip()
                if txt and lbl in LABEL_TO_ID:
                    texts.append(txt)
                    labels.append(LABEL_TO_ID[lbl])
    return texts, labels

def stratified_split(texts: List[str], labels: List[int], val_ratio: float = 0.15, seed: int = 42):
    random.seed(seed)
    class_groups = {0: [], 1: [], 2: []}
    for t, l in zip(texts, labels):
        class_groups[l].append(t)

    train_texts, train_labels = [], []
    val_texts, val_labels = [], []

    for l, group in class_groups.items():
        random.shuffle(group)
        val_count = max(1, int(len(group) * val_ratio))
        val_texts.extend(group[:val_count])
        val_labels.extend([l] * val_count)
        train_texts.extend(group[val_count:])
        train_labels.extend([l] * (len(group) - val_count))

    # Shuffle training items
    combined = list(zip(train_texts, train_labels))
    random.shuffle(combined)
    train_texts, train_labels = zip(*combined)

    return list(train_texts), list(train_labels), val_texts, val_labels

def compute_metrics(preds: List[int], targets: List[int]) -> Dict[str, float]:
    total = len(targets)
    if total == 0:
        return {"accuracy": 0.0, "f1": 0.0}
    correct = sum(1 for p, y in zip(preds, targets) if p == y)
    acc = correct / total

    # Weighted F1
    f1_weighted = 0.0
    for c in [0, 1, 2]:
        tp = sum(1 for p, y in zip(preds, targets) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, targets) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, targets) if p != c and y == c)
        support = sum(1 for y in targets if y == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_class = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_weighted += f1_class * (support / total)

    return {"accuracy": round(acc, 4), "f1": round(f1_weighted, 4)}

def train_rtx5050(
    data_path: str = "data/master_train_dataset.csv",
    base_model_name: str = "l3cube-pune/hing-bert",
    output_dir: str = "models/hing_hate_bert_rtx5050",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5
):
    print("=" * 70)
    print("DEVIPUTRA AI // HINGLISH TRANSFORMER FINE-TUNING PIPELINE")
    print("=" * 70)

    # 1. Device and Architecture Detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device = torch.device("cuda:0")
        print(f"[GPU ENGINE ACTIVE]: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[DEVICE]: CPU (Optimized for lightweight inference)")

    # 2. Load Dataset using pure csv module
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "sample.csv")
        
    print(f"Loading Hinglish training corpus: {data_path}...")
    texts, labels = load_data_clean(data_path)
    print(f"Total samples: {len(texts)} samples loaded.")

    train_texts, train_labels, val_texts, val_labels = stratified_split(texts, labels, val_ratio=0.15)
    print(f"Split: {len(train_texts)} train items | {len(val_texts)} validation items.")

    # 3. Load HuggingFace Tokenizer & Model
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.optim import AdamW

    print(f"Loading transformer backbone ({base_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )
    model.to(device)

    train_ds = HinglishTextDataset(train_texts, train_labels, tokenizer)
    val_ds = HinglishTextDataset(val_texts, val_labels, tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_f1 = 0.0
    os.makedirs(output_dir, exist_ok=True)

    print("\nSTARTING TRAINING LOOPS...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(f"   Epoch [{epoch}/{epochs}] | Step [{step+1}/{len(train_loader)}] | Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_labels = batch['label']

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
                
                val_preds.extend(preds)
                val_targets.extend(target_labels.tolist())

        metrics = compute_metrics(val_preds, val_targets)
        val_acc = metrics["accuracy"]
        val_f1 = metrics["f1"]

        print(f"\nEPOCH {epoch} SUMMARY:")
        print(f"   Train Loss     : {avg_train_loss:.4f}")
        print(f"   Validation Acc : {val_acc * 100:.2f}%")
        print(f"   Validation F1  : {val_f1:.4f}")
        print("-" * 50)

        # Save Best Checkpoint
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            print(f"Saving checkpoint to: '{output_dir}'...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print("\nTRAINING COMPLETE!")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hinglish Fine-Tuning Script")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_rtx5050(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
