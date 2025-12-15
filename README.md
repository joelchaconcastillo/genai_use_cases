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

