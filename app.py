from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize the FastAPI app
app = FastAPI()

# Load model and tokenizer (assuming model artifacts are saved in `models/text_classifier`)
model = BertForSequenceClassification.from_pretrained('models/text_classifier')
tokenizer = BertTokenizer.from_pretrained('models/text_classifier')

# Define a request body class
class PredictionRequest(BaseModel):
    text: str

# Define the prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return {"prediction": "hate" if predicted_class == 1 else "noHate"}
