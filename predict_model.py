import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('models/text_classifier')
tokenizer = BertTokenizer.from_pretrained('models/text_classifier')

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return "hate" if predicted_class == 1 else "noHate"

if __name__ == "__main__":
    sample_text = input("Enter text to classify: ")
    prediction = predict(sample_text)
    print(f"Prediction: {prediction}")
