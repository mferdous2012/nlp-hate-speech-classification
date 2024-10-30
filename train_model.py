import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

# Set up paths
data_dir = 'data/hate-speech-dataset'
train_dir = os.path.join(data_dir, 'sampled_train')
test_dir = os.path.join(data_dir, 'sampled_test')
annotations_file = os.path.join(data_dir, 'annotations_metadata.csv')
model_dir = 'models/text_classifier'
results_dir = 'results'

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Load annotations and apply label mapping
annotations = pd.read_csv(annotations_file)
print("Sample of annotations data:", annotations.head())  # Debugging: view sample data

label_mapping = {'hate': 1, 'noHate': 0}
annotations['label'] = annotations['label'].map(label_mapping)

# Separate train and test annotations based on file presence in directories
train_file_ids = set(f.replace('.txt', '') for f in os.listdir(train_dir))
test_file_ids = set(f.replace('.txt', '') for f in os.listdir(test_dir))

train_annotations = annotations[annotations['file_id'].isin(train_file_ids)]
test_annotations = annotations[annotations['file_id'].isin(test_file_ids)]

# Helper function to load text data
def load_texts(data_dir, file_ids):
    texts = []
    for file_id in file_ids:
        file_path = os.path.join(data_dir, f"{file_id}.txt")  # Append .txt to each file_id
        if os.path.exists(file_path):  # Check if file exists
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
        else:
            print(f"Warning: File {file_path} not found.")  # Debugging: log missing files
    return texts

# Load training and testing data
train_texts = load_texts(train_dir, train_annotations['file_id'])
train_labels = train_annotations['label'].tolist()

test_texts = load_texts(test_dir, test_annotations['file_id'])
test_labels = test_annotations['label'].tolist()

# Check if data was loaded correctly
if len(train_texts) == 0 or len(train_labels) == 0:
    raise ValueError("No training data found. Please ensure the 'sampled_train' directory and file paths are correct.")
if len(test_texts) == 0 or len(test_labels) == 0:
    raise ValueError("No testing data found. Please ensure the 'sampled_test' directory and file paths are correct.")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length)
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Create datasets and dataloaders
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TextDataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize model and optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(3):  # Adjust the number of epochs as needed
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save model and tokenizer
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Evaluation on test set
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.tolist())
        true_labels.extend(batch['labels'].tolist())

# Calculate metrics and save results
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, target_names=['noHate', 'hate'], output_dict=True)
results_df = pd.DataFrame(report).transpose()
results_df['accuracy'] = accuracy
results_df.to_csv(os.path.join(results_dir, 'test_results.csv'), index=True)

print("Model training and evaluation completed.")
print(f"Accuracy: {accuracy}")
print("Detailed results saved in results/test_results.csv")
