# Semantic Quote Retrieval System

A semantic search engine that retrieves contextually similar quotes using transformer-based embeddings and FAISS for efficient vector similarity search.

---

## 🚀 Overview

This project implements an end-to-end semantic retrieval pipeline that enables meaning-based search over a large corpus of quotes (~500K entries). Unlike traditional keyword search, this system uses dense vector embeddings to capture semantic similarity between sentences.

Given an input sentence, the system returns the top-k most semantically similar quotes, even if the wording differs.

---

## 🧠 How It Works

### Offline Phase (Index Construction)

1. **Tokenization**
   - Convert text into token IDs using a pretrained tokenizer.

2. **Embedding Generation**
   - Use a pretrained transformer-based embedding model (`embeddinggemma-300m`) to generate token-level embeddings.

3. **Pooling**
   - Apply mean pooling across tokens to obtain a single vector representation per sentence.

4. **Normalization**
   - Normalize embeddings to enable cosine similarity using inner product.

5. **Indexing (FAISS)**
   - Store vectors in a FAISS index for efficient nearest neighbor search.

6. **Metadata Storage**
   - Store `(quote, author)` pairs in a SQLite database mapped by index.

---

### Online Phase (Querying)

1. Input sentence is tokenized and embedded using the same pipeline.
2. FAISS retrieves the top-k nearest vectors.
3. Corresponding quotes are fetched from the SQLite database.
4. Results are returned with similarity scores.

---

## 🏗️ Architecture
```
Offline:
quotes → tokenize → embed → mean pool → normalize → FAISS index
↓
SQLite DB (quote, author)

Online:
query → tokenize → embed → mean pool → normalize
↓
FAISS search → retrieve indices → lookup quotes
```

---

## ⚙️ Tech Stack

- **JAX / Flax** — Model inference
- **Transformers (Hugging Face)** — Tokenization
- **FAISS** — Vector similarity search
- **SQLite** — Persistent key-value storage
- **NumPy** — Data handling

---

## 📦 Project Structure
```
semantic-quote-search/
│
├── README.md
├── environment.yml
├── .gitignore
│
├── quotes.csv.gz              # raw dataset
├── input.txt                  # query inputs, (for input quotes)
│
├── make_index.py              # builds embeddings + FAISS index
├── find_quote.py              # query + retrieval pipeline
├── helper.py                  # model loading utilities
│
├── bonsai/
│   ├── gemma3/
│   │   ├── modeling.py        # embedding model implementation
│   │   └── params.py
│   └── utils/
│       ├── params.py
│       ├── rope.py
│       └── samplers.py
│
├── job_gpu_make_index.sh      # PACE job for index building
└── job_gpu_find_quote.sh      # PACE job for querying
```

---

## 🛠️ Setup

### 1. Create environment

```
conda env create -f environment.yml
conda activate cs3600-llm
```

---

### 2. Install additional dependencies (if needed)
```
pip install faiss-cpu torch flax safetensors huggingface_hub transformers
```

---

## 🔑 Hugging Face Access
This project uses a gated model:

👉 https://huggingface.co/google/embeddinggemma-300m

Steps:
1.	Request access and accept terms
2.	Create an access token
3.	Set environment variable:
```
export HF_TOKEN="your_token_here"
```

---

## 📥 Dataset

Download the dataset from the link below:

👉 https://drive.google.com/file/d/1FRVA5LMqi1b5V8hp00d-LWu993f7u6JV/view?usp=sharing

After downloading:

1. Move the file into the project root directory
2. Ensure the filename is: _quotes.csv.gz_

---

### ▶️ Usage
## Step 1: Build index
```
python make_index.py
```
Outputs:
- quotes.index
- quotes.db

## Step 2: Run search
Add queries to input.txt, then:
```
python find_quote.py
```
Outputs:
- Console results (top matches)
- recent_found_indices.txt

---

### 🧪 Example
Input:
```
Why is a carrot more orange than an orange?
```
Output:
```
218163,0.781: '...'
129997,0.778: '...'
413215,0.776: '...'
```

---

### 📊 Key Features
- Semantic (meaning-based) search
- Fast nearest neighbor retrieval (FAISS)
- Scales to hundreds of thousands of entries
- Supports flexible natural language queries

---

### ⚠️ Notes
- Full dataset (~500K quotes) may take time to process locally
- GPU environments (e.g., HPC clusters) can significantly speed up embedding generation
- SQLite is used instead of dbm for better scalability

---

### 📚 References
- FAISS: https://github.com/facebookresearch/faiss
- Hugging Face Transformers: https://huggingface.co
- JAX: https://github.com/google/jax

---

### 👤 Author
Khoa Bui
