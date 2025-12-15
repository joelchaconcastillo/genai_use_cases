# GenAI Hands-on Cheatsheet

A concise, practical cheatsheet with runnable code snippets to help you prepare for GenAI engineering tasks listed in your Statement of Work.

**Setup**
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Quick: Model inference (Hugging Face Transformers)**

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
device = 0 if torch.cuda.is_available() else -1

pipe = pipeline("text-generation", model=model_name, device=device)
resp = pipe("Translate to Spanish: How are you?", max_new_tokens=40)
print(resp[0]["generated_text"]) 
```

Notes: replace `gpt2` with a local or Hugging Face Hub model appropriate for your task.

**Fine-tuning (Hugging Face Trainer) — minimal text classification**

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./out", num_train_epochs=1, per_device_train_batch_size=8, logging_steps=50
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"].select(range(2000)), eval_dataset=dataset["test"].select(range(500)))
trainer.train()
```

**Embeddings + FAISS (semantic retrieval)**

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = ["Hello world","Generative AI patterns","How to fine-tune models"]
embs = model.encode(docs, convert_to_numpy=True)

dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)

q = model.encode(["fine-tuning steps"], convert_to_numpy=True)
distances, idxs = index.search(q, k=2)
print(idxs, distances)
```

**RAG (retrieval-augmented generation) — sketch with LangChain**

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Prepare embeddings and index (example with OpenAI)
emb = OpenAIEmbeddings()
# texts = [ ... ]
# embeddings = emb.embed_documents(texts)
# faiss_index = FAISS.from_documents(texts, emb)

# Use RetrievalQA
qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-4o-mini"), chain_type="stuff", retriever=faiss_index.as_retriever())
print(qa.run("Summarize the approach for retrieval-augmented generation."))
```

Replace `OpenAI` usage with your preferred LLM provider and set API keys through environment variables.

**Serving a model: FastAPI example**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
generator = pipeline('text-generation', model='gpt2')

class Req(BaseModel):
    prompt: str

@app.post('/generate')
def generate(req: Req):
    out = generator(req.prompt, max_new_tokens=80)
    return {"text": out[0]['generated_text']}

# Run: uvicorn app:app --reload --port 8000
```

**Evaluation snippets**

```python
from sklearn.metrics import accuracy_score, f1_score
# Binary classification
y_true = [0,1,1,0]
y_pred = [0,1,0,0]
print('Accuracy', accuracy_score(y_true,y_pred))
print('F1', f1_score(y_true,y_pred))
```

**Testing a small function**

```python
def normalize(s):
    return s.strip().lower()

def test_normalize():
    assert normalize(' Hello ') == 'hello'

if __name__ == '__main__':
    test_normalize(); print('ok')
```

**Data Privacy & Security Checklist**
- Minimize and anonymize PII in training data.
- Prefer embeddings + retrieval over sending raw docs to LLM for sensitive info.
- Use encryption-at-rest for indexes and storage.
- Apply access controls to API keys and rotate them regularly.
- Keep audit logs of data access and model outputs when required.

**Useful commands**
- Format Python: `python -m pip install black && black .`
- Run FastAPI server: `uvicorn app:app --reload --port 8000`

---
If you want, I can expand any section into a full example folder (notebooks, scripts, tests, and CI). Tell me which topics you want prioritized.
