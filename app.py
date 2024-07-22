import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Define the path to the model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pth')

# Load the tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define a simple model class


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        return self.fc(x)


# Load the simple model
simple_model = SimpleModel()
simple_model.load_state_dict(torch.load(
    model_path, map_location=torch.device('cpu')))
simple_model.eval()

# Processes the text and returns the embeddings


def get_embedding(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=512)
    # Get embeddings from BERT model
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the mean of the last hidden state as the embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


@app.route('/match', methods=['POST'])
def match():
    data = request.json
    resume_text = data['resume']
    job_description_text = data['job_description']

    # Get embeddings for resume and job description
    resume_embedding = get_embedding(resume_text)
    job_description_embedding = get_embedding(job_description_text)

    # Use the simple model to process the embeddings
    resume_embedding_processed = simple_model(resume_embedding).squeeze()
    job_description_embedding_processed = simple_model(
        job_description_embedding).squeeze()

    # Calculate similarity between processed embeddings
    similarity = torch.nn.functional.cosine_similarity(
        resume_embedding_processed, job_description_embedding_processed, dim=0)
    compatibility_score = similarity.item()

    return jsonify({'compatibility_score': compatibility_score})


if __name__ == '__main__':
    app.run(debug=True)
