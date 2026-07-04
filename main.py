import os
# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Constants
MODEL_NAME = "distilbert-base-uncased"
LABELS = ["Positive", "Negative", "Neutral"]

# Custom sentiment dataset
TRAIN_DATA = [
    # Positive (0)
    ("This product exceeded all my expectations, absolutely fantastic!", 0),
    ("Excellent customer support, very friendly and fast resolution.", 0),
    ("I love the clean interface and the incredibly fast performance.", 0),
    ("Highly recommended for anyone looking for a reliable solution.", 0),
    ("Amazing speed and very intuitive user experience overall.", 0),
    
    # Negative (1)
    ("Extremely disappointed with the service, it was slow and buggy.", 1),
    ("The application crashes constantly and is completely unusable.", 1),
    ("Terrible technical support, they did not help me at all.", 1),
    ("The interface is confusing and the loading times are awful.", 1),
    ("A complete waste of money, I do not recommend it to anyone.", 1),
    
    # Neutral (2)
    ("The product is okay, it performs exactly as advertised.", 2),
    ("Average performance, nothing special but works fine.", 2),
    ("It is a standard utility that gets the job done.", 2),
    ("An acceptable experience, though there is room for improvement.", 2),
    ("The speed is moderate and the features are standard.", 2)
]

VAL_DATA = [
    ("The customer experience was superb and the output is great.", 0),
    ("Awesome framework that makes development so much easier.", 0),
    ("This app is slow, unresponsive, and full of glitches.", 1),
    ("Very poor quality and highly overpriced, avoid at all costs.", 1),
    ("The software is decent, it works as expected.", 2),
    ("A normal tool with basic functionalities and simple design.", 2)
]

# PyTorch Dataset Subclass
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Metrics evaluation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("=" * 70)
    print("Project 83: Custom Text Classifier with HuggingFace Trainer API")
    print("=" * 70)

    # 1. Tokenize Data
    print("[*] Loading pre-trained DistilBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("[*] Encoding training and validation text sequences...")
    train_texts = [item[0] for item in TRAIN_DATA]
    train_labels = [item[1] for item in TRAIN_DATA]
    val_texts = [item[0] for item in VAL_DATA]
    val_labels = [item[1] for item in VAL_DATA]
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=32)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=32)
    
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    print(f"[+] Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples.\n")

    # 2. Load Pretrained PyTorch Classification Model
    print("[*] Loading pre-trained DistilBERT classifier model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, use_safetensors=False
    )
    print()

    # 3. Setup Trainer
    print("[*] Configuring Hugging Face TrainingArguments...")
    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=8,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        disable_tqdm=True  # Keeps stdout clean and prints metrics linearly
    )

    print("[*] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 4. Train Model
    print("\n[*] Starting Trainer API optimization loop...")
    start_time = time.time()
    trainer.train()
    training_duration = time.time() - start_time
    print(f"\n[+] Fine-tuning completed in {training_duration:.2f} seconds.\n")

    # 5. Evaluate Validation Set
    print("[*] Running final evaluations on validation set...")
    eval_metrics = trainer.evaluate()
    print("\nValidation Metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k:15s}: {v}")
    print()

    # Compute validation predictions and confusion matrix
    val_predictions_out = trainer.predict(val_dataset)
    val_preds = val_predictions_out.predictions.argmax(-1)
    conf_matrix = confusion_matrix(val_labels, val_preds)

    # 6. Unseen Out-of-Domain Generalization Test
    test_queries = [
        "This is the best service I have ever used! Super fast and reliable.",
        "The system is completely broken and crashes every time I open it.",
        "The speed is average, not too slow and not too fast.",
        "The package arrived on time and was in standard packaging."
    ]
    
    print("[*] Predicting sentiment for unseen out-of-domain queries:")
    test_encodings = tokenizer(test_queries, truncation=True, padding=True, max_length=32)
    # Dummy labels for dataset format compatibility
    test_dataset = CustomDataset(test_encodings, [0] * len(test_queries))
    
    query_predictions_out = trainer.predict(test_dataset)
    query_logits = query_predictions_out.predictions
    
    # Softmax to get probabilities
    query_probs = np.exp(query_logits) / np.sum(np.exp(query_logits), axis=-1, keepdims=True)
    query_pred_classes = query_probs.argmax(-1)
    
    query_reports = []
    for idx, (query, pred_idx) in enumerate(zip(test_queries, query_pred_classes)):
        conf = query_probs[idx][pred_idx]
        pred_label = LABELS[pred_idx]
        query_reports.append({
            "text": query,
            "prediction": pred_label,
            "confidence": conf
        })
        print(f"  Query: \"{query}\"\n    -> Predicted: {pred_label:10s} (Confidence: {conf*100:.2f}%)")
    print()

    # 7. Extract Loss and Metric logs from Trainer History
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    val_f1_history = []
    
    for log in trainer.state.log_history:
        if 'loss' in log:
            train_loss_history.append((log['epoch'], log['loss']))
        if 'eval_loss' in log:
            val_loss_history.append((log['epoch'], log['eval_loss']))
            val_acc_history.append((log['epoch'], log['eval_accuracy']))
            val_f1_history.append((log['epoch'], log['eval_f1']))

    # 8. Generate 2x2 Visual Dashboard
    print("[*] Generating dashboard...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14), facecolor='#0F172A')
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28)

    # Panel 1: Loss Curves (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1E293B')
    if train_loss_history:
        t_epochs, t_losses = zip(*train_loss_history)
        ax1.plot(t_epochs, t_losses, label='Training Loss', color='#38BDF8', linewidth=2.0, marker='o')
    if val_loss_history:
        v_epochs, v_losses = zip(*val_loss_history)
        ax1.plot(v_epochs, v_losses, label='Validation Loss', color='#F59E0B', linewidth=2.0, marker='s')
    ax1.set_xlabel('Epochs', color='#94A3B8', fontsize=11)
    ax1.set_ylabel('Loss', color='#94A3B8', fontsize=11)
    ax1.tick_params(colors='#94A3B8')
    ax1.grid(True, linestyle=':', color='#334155', alpha=0.5)
    ax1.legend(facecolor='#0F172A', edgecolor='#334155', fontsize=10)
    ax1.set_title("Trainer Optimization Loss Curves", color='#38BDF8', fontsize=15, fontweight='bold', pad=15)

    # Panel 2: Metric Curves (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1E293B')
    if val_acc_history:
        acc_epochs, acc_vals = zip(*val_acc_history)
        ax2.plot(acc_epochs, [a * 100 for a in acc_vals], label='Val Accuracy', color='#10B981', linewidth=2.0, marker='o')
    if val_f1_history:
        f1_epochs, f1_vals = zip(*val_f1_history)
        ax2.plot(f1_epochs, [f * 100 for f in f1_vals], label='Val Macro F1', color='#EC4899', linewidth=2.0, marker='s')
    ax2.set_xlabel('Epochs', color='#94A3B8', fontsize=11)
    ax2.set_ylabel('Score (%)', color='#94A3B8', fontsize=11)
    ax2.tick_params(colors='#94A3B8')
    ax2.grid(True, linestyle=':', color='#334155', alpha=0.5)
    ax2.legend(facecolor='#0F172A', edgecolor='#334155', fontsize=10)
    ax2.set_title("Validation Classification Metrics", color='#10B981', fontsize=15, fontweight='bold', pad=15)

    # Panel 3: Confusion Matrix Heatmap (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1E293B')
    im3 = ax3.imshow(conf_matrix, cmap="Blues", aspect='auto')
    cbar3 = fig.colorbar(im3, ax=ax3, pad=0.03, shrink=0.85)
    cbar3.ax.tick_params(colors='#94A3B8')
    
    ticks = np.arange(len(LABELS))
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(LABELS, color='#94A3B8', fontsize=10)
    ax3.set_yticks(ticks)
    ax3.set_yticklabels(LABELS, color='#94A3B8', fontsize=10)
    
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            score = conf_matrix[i, j]
            text_color = "black" if score > (np.max(conf_matrix) / 2) else "white"
            ax3.text(j, i, f"{score}", ha="center", va="center", color=text_color, fontweight='bold', fontsize=14)
            
    ax3.set_xlabel("Predicted Sentiment", color='#94A3B8', fontsize=11, labelpad=8)
    ax3.set_ylabel("True Sentiment", color='#94A3B8', fontsize=11, labelpad=8)
    ax3.tick_params(colors='#94A3B8')
    ax3.set_title("HuggingFace Trainer Confusion Matrix", color='#F59E0B', fontsize=15, pad=15, fontweight='bold')

    # Panel 4: Out-of-Domain Generalization Report Card (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1E293B')
    ax4.axis('off')
    card4 = plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax4.transAxes, facecolor='#1E293B', edgecolor='#334155', linewidth=1.5)
    ax4.add_patch(card4)
    
    ax4.text(0.06, 0.88, "OUT-OF-DOMAIN GENERALIZATION REPORT", color='#EC4899', fontsize=12, fontweight='bold')
    
    y_offset = 0.72
    colors_dict = {
        "Positive": '#10B981',
        "Negative": '#EF4444',
        "Neutral": '#F59E0B'
    }
    
    for idx, rep in enumerate(query_reports):
        q_wrapped = rep["text"]
        if len(q_wrapped) > 65:
            q_wrapped = q_wrapped[:62] + "..."
            
        color_pred = colors_dict[rep['prediction']]
        ax4.text(0.08, y_offset, f"Query: \"{q_wrapped}\"", color='#E2E8F0', fontsize=10, fontfamily='monospace')
        ax4.text(0.08, y_offset - 0.05, f"➔ Predicted: {rep['prediction'].upper():10s}   |   Softmax Confidence: {rep['confidence']*100:.2f}%", 
                 color=color_pred, fontsize=10.5, fontweight='bold')
        
        if idx < len(query_reports) - 1:
            ax4.plot([0.08, 0.92], [y_offset - 0.08, y_offset - 0.08], color='#334155', linestyle='-', linewidth=1.0)
            
        y_offset -= 0.19

    ax4.set_title("Filter Inference & Generalization Report", color='#EC4899', fontsize=15, pad=15, fontweight='bold')

    fig.suptitle("Project 83: Custom Text Classifier with HF Trainer API", fontsize=24, fontweight='bold', color='#E2E8F0', y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "trainer_classification_results.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    
    print(f"\n[+] Fine-tuned classifier dashboard saved to '{save_path}'")
    print("=" * 70)

if __name__ == "__main__":
    main()
