# 🧠 GenAI Cookbook

A practical, notebook-based collection of **real-world Generative AI patterns and systems**.  
This repository contains **self-contained Jupyter notebooks** that demonstrate how to design, build, and orchestrate modern GenAI applications — from RAG pipelines to multi-agent and multimodal systems.
---

## 🎯 Goals

- Provide clear, runnable examples of common Generative AI capabilities
- Focus on reusable **design patterns**, not just libraries or tools
- Progress from fundamentals to advanced orchestration
- Serve as a long-term **reference cookbook** for GenAI system builders

---

## 🗂 Repository Structure

The repository is organized by **capability first**, then **pattern**, and finally **tool-specific implementations**.


genai-cookbook/
├─ notebooks/
│ ├─ 01-fundamentals/
│ ├─ 02-embeddings/
│ ├─ 03-retrieval-and-search/
│ ├─ 04-data-reasoning/
 │ ├─ 05-multimodal/
│ ├─ 06-conversational-systems/
│ ├─ 07-agents/
│ ├─ 08-orchestration-patterns/
│ ├─ 09-langgraph-patterns/
│ ├─ 10-end-to-end-systems/
├─ data/
├─ utils/
└─ README.md


---

## 📚 Topics Covered

### 1️⃣ Fundamentals
Core concepts required for all GenAI systems.

- Prompt engineering
- Model parameters & determinism
- Prompt templates and system prompts

---

### 2️⃣ Embeddings & Vector Search
Foundations for semantic retrieval.

- Embedding generation
- Chunking strategies
- Vector similarity search

---

### 3️⃣ Retrieval & Search Systems
Getting the right information into the model.

- Retrieval-Augmented Generation (RAG)
- Recommender systems
- Web search with Tavily
- Hybrid retrieval strategies

---

### 4️⃣ Data Reasoning & Querying
LLMs reasoning over structured data.

- DataFrame analyzers
- Natural language to SQL
- Query generation & execution

---

### 5️⃣ Multimodal AI
Beyond text-only workflows.

- Multimodal LLMs
- Image caption generation
- Text-to-Speech (TTS)
- Speech-to-Text (STT)

---

### 6️⃣ Conversational & UI Systems
User-facing AI applications.

- AI-powered dashboards
- Conversational systems
- BeeAI-based AI conversations

---

### 7️⃣ Agent Systems
Autonomous and tool-using LLMs.

- Single-agent workflows
- Multi-agent systems
- CrewAI fundamentals
- Custom CrewAI tools and classes

---

### 8️⃣ Orchestration Patterns
Controlling execution flow and reasoning.

- Orchestrator pattern
- Evaluator–Optimizer pattern
- Tool selection and retry strategies

---

### 9️⃣ LangGraph Patterns
Explicit control-flow graphs for LLM systems.

- Sequence pattern
- Routing pattern
- Parallelization pattern

---

### 🔟 End-to-End Systems
Full systems combining multiple capabilities.

- RAG + agents + orchestration
- Multimodal assistants
- Data-aware AI dashboards

---

## 🧩 Notebook Metadata Convention

Each notebook starts with a short header:

```md
**Capability:** Retrieval & Search  
**Pattern:** Orchestrator / Routing  
**Tools:** LangGraph, OpenAI, FAISS

```

---

## 🛠 Hands-on: How to use this repo (practical workflow)

This repository is organized as a cookbook of patterns. To turn these patterns into functional source code and portfolio pieces, follow this practical workflow:

- **1. Read the Cheatsheet:** start with `CHEATSHEET.md` for quick, runnable examples and setup instructions.
- **2. Pick a priority use case:** choose one of the topics below that maps to your contractor duties (e.g., RAG demo, embedding pipeline, fine-tuning classifier).
- **3. Create an example folder:** under a directory such as `examples/<topic>-<short-name>/` include:
  - `notebook.ipynb` — an end-to-end demo and narrative.
  - `app.py` or `service/` — minimal API server (FastAPI) to expose the model.
  - `requirements.txt` — pinned deps for reproducibility.
  - `README.md` — how to run, dataset pointers, and expected outputs.
  - `tests/` — small unit tests (sanity checks and smoke tests).
- **4. Iteratively improve:** add evaluation, logging, privacy notes, and CI workflow.

**Centralized configuration & LLM management**

This repo uses a small centralized configuration pattern to make examples portable and reproducible:

- Store named LLM profiles in `configs/llm_profiles.yaml` (example included).
- Put secrets in environment variables and an optional local `.env` (see `.env.example`).
- Use `utils/llm_manager.py` to load profiles and get a small client with `generate(prompt)`.

This approach keeps secrets out of code, lets examples pick named profiles, and makes it easy to switch providers.

## 🎯 Prioritized examples (aligns to your Statement of Work)

- **RAG demo (High priority):** embeddings -> FAISS -> retriever -> LLM summarization. Good for business-aligned extraction tasks.
- **Embedding pipeline (High priority):** dataset chunking, embedding generation, index storage/encryption.
- **Fine-tuning prototype (Medium):** small-scope fine-tune (classification or summarization) with evaluation metrics and reproducible training script.
- **Serving & Integrations (Medium):** FastAPI service + minimal frontend or a demo curl workflow.
- **Data reasoning / SQL assistant (Medium):** natural-language-to-SQL with safety checks and execution sandboxing.
- **Multimodal prototype (Low/Optional):** image captioning or TTS demo.

## ✅ Repo conventions for example folders

- Use `examples/<topic>-<short-name>/` as the canonical place for runnable code.
- Each example MUST include `README.md` with these headings: Purpose, Setup, Run, Expected output, Notes on Privacy & Limitations.
- Keep scripts idempotent and small (one CLI per script).
- Add a `requirements.txt` and, when possible, `environment.yml` or `pyproject.toml` for reproducible installs.

## 🔐 Data privacy & security (baseline checklist)

- Avoid including raw PII in repo assets.
- Document any sensitive data requirements in the example `README.md` and suggest synthetic datasets for demos.
- Provide instructions for storing API keys in environment variables (never hardcode secrets).

---

## GenAI Hands-on Cheatsheet

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

## ✅ Repo conventions for example folders

- Use `examples/<topic>-<short-name>/` as the canonical place for runnable code.
- Each example MUST include `README.md` with these headings: Purpose, Setup, Run, Expected output, Notes on Privacy & Limitations.
- Keep scripts idempotent and small (one CLI per script).
- Add a `requirements.txt` and, when possible, `environment.yml` or `pyproject.toml` for reproducible installs.

## 🔐 Data privacy & security (baseline checklist)

- Avoid including raw PII in repo assets.
- Document any sensitive data requirements in the example `README.md` and suggest synthetic datasets for demos.
- Provide instructions for storing API keys in environment variables (never hardcode secrets).

## 🧭 Next steps I can do for you

- Scaffold an `examples/rag-demo/` with notebook, scripts, and tests.
- Create a runnable FastAPI service with Dockerfile and simple CI workflow that runs smoke tests.
- Expand `CHEATSHEET.md` into a notebook and small example repo for one chosen topic.

Tell me which next step you want prioritized and I will scaffold it (RAG demo, Fine-tuning prototype, or FastAPI service).



## 📚 Topics Covered

### 1️⃣ Fundamentals
Core concepts required for all GenAI systems.

- Prompt engineering
- Model parameters & determinism
- Prompt templates and system prompts

---

### 2️⃣ Embeddings & Vector Search
Foundations for semantic retrieval.

- Embedding generation
- Chunking strategies
- Vector similarity search

---

### 3️⃣ Retrieval & Search Systems
Getting the right information into the model.

- Retrieval-Augmented Generation (RAG)
- Recommender systems
- Web search with Tavily
- Hybrid retrieval strategies

---

### 4️⃣ Data Reasoning & Querying
LLMs reasoning over structured data.

- DataFrame analyzers
- Natural language to SQL
- Query generation & execution

---

### 5️⃣ Multimodal AI
Beyond text-only workflows.

- Multimodal LLMs
- Image caption generation
- Text-to-Speech (TTS)
- Speech-to-Text (STT)

---

### 6️⃣ Conversational & UI Systems
User-facing AI applications.

- AI-powered dashboards
- Conversational systems
- BeeAI-based AI conversations

---

### 7️⃣ Agent Systems
Autonomous and tool-using LLMs.

- Single-agent workflows
- Multi-agent systems
- CrewAI fundamentals
- Custom CrewAI tools and classes

---

### 8️⃣ Orchestration Patterns
Controlling execution flow and reasoning.

- Orchestrator pattern
- Evaluator–Optimizer pattern
- Tool selection and retry strategies

---

### 9️⃣ LangGraph Patterns
Explicit control-flow graphs for LLM systems.

- Sequence pattern
- Routing pattern
- Parallelization pattern

---

### 🔟 End-to-End Systems
Full systems combining multiple capabilities.

- RAG + agents + orchestration
- Multimodal assistants
- Data-aware AI dashboards

---

## 🧩 Notebook Metadata Convention

Each notebook starts with a short header:

```md
**Capability:** Retrieval & Search  
**Pattern:** Orchestrator / Routing  
**Tools:** LangGraph, OpenAI, FAISS

