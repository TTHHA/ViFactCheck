# Import required libraries
import warnings
import logging
import torch
import pandas as pd
import ast
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import classification_report
import transformers

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
transformers.logging.set_verbosity_error()

# Define SentencePairDataset class
class SentencePairDataset(Dataset):
    def __init__(self, sentence_pairs, labels, tokenizer, max_length):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sentence1, sentence2, len_e = self.sentence_pairs[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            sentence1,
            text_pair=sentence2,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Define PhoBERTClassifier class
class PhoBERTClassifier(nn.Module):
    def __init__(self, phobert, num_classes):
        super(PhoBERTClassifier, self).__init__()
        self.phobert = phobert
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.phobert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        dropout_output = self.dropout(pooled_output)
        logits = self.linear(dropout_output)
        return logits

    def push_to_hub(self, name, token):
        self.phobert.push_to_hub(name, token)

# Define helper functions
def convert_to_list(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def join_sentences(evidences):
    return ' '.join([sentence if sentence.endswith('.') else sentence + '.' for sentence in evidences[:3]])

def count_len(table):
    table["Evidence_Merge_BM25"] = ""
    for i in range(len(table)):
        table["Evidence_Merge_BM25"][i] = join_sentences(table['evidence_top5'][i])
    return table

# Load and preprocess data
train_data = pd.read_csv("./train_sbert.csv")
test_data = pd.read_csv("./test_sbert.csv")
dev_data = pd.read_csv("./dev_sbert.csv")

for data in [train_data, test_data, dev_data]:
    data['evidence_top5'] = data['evidence_top5'].apply(convert_to_list)
    data['Evidence_List'] = data['Evidence_List'].apply(convert_to_list)
    data = count_len(data)

# Prepare datasets
def prepare_dataset(data):
    statements = list(data["Statement"])
    evidences = list(data["Evidence_Merge_BM25"])
    len_evidences = list(data["len_evidence"])
    return [(s, e, l) for s, e, l in zip(statements, evidences, len_evidences)]

X_train = prepare_dataset(train_data)
X_test = prepare_dataset(test_data)
X_dev = prepare_dataset(dev_data)

y_train = list(train_data["labels"])
y_test = list(test_data["labels"])
y_dev = list(dev_data["labels"])

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
phobert = AutoModel.from_pretrained("vinai/phobert-large")
max_length = 256

# Prepare datasets and dataloaders
train_dataset = SentencePairDataset(X_train, y_train, tokenizer, max_length)
dev_dataset = SentencePairDataset(X_dev, y_dev, tokenizer, max_length)
test_dataset = SentencePairDataset(X_test, y_test, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=16)
dev_loader = DataLoader(dev_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Set up device, model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhoBERTClassifier(phobert, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.5e-5)

# Training loop
from tqdm import tqdm

epochs = 10

for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    predictions = []
    true_labels = []
    for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{epochs} - Evaluation"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    print(f"Epoch {epoch+1}/{epochs}")
    print(classification_report(true_labels, predictions, digits=4))