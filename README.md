# NLP Text Classification Project

## Goal
The aim of this project is to develop a text classification model that can identify hate speech using FastAPI. The project follows these key steps:

1. **Data Preparation**:
   - Text data and corresponding labels are loaded from a dataset containing `.txt` files.
   - Preprocessing includes tokenization using BERT.

2. **Model Training**:
   - A BERT-based model (`bert-base-uncased`) is fine-tuned on the training data.
   - The model distinguishes between `hate` and `noHate` categories.

3. **Model Deployment**:
   - A FastAPI application serves the model as an API with a `/predict` endpoint.
   - The API expects a JSON input with a text field and returns the classification result.

## API Usage
To interact with the API, send a POST request to `/predict` with JSON input:

```json
{
    "text": "Sample text to classify."
}
