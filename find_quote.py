import sqlite3
import faiss
import pickle
from time import perf_counter
import os
import numpy as np
import jax.numpy as jnp
import jax
from transformers import AutoTokenizer
from helper import load_model

DB_PATH = "quotes.db"
INDEX_PATH = "quotes.index"
# How many quotes should we display?
BEST_K = 3

print("Using devices:", jax.devices())
device = jax.devices()[0]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(
    "google/embeddinggemma-300m",
    token=os.environ["HF_TOKEN"]
)
model = load_model()

print("Loading index...")
index = faiss.read_index(INDEX_PATH)

# Open the key-value store
print("Using key-value store of type sqlite3")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

results_file = open("recent_found_indices.txt", "w")

# Read quotes from input.txt and process each one
with open("input.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for line_num, sample_quote in enumerate(lines, start=1):
    print("=" * 60)
    print(f"Processing line {line_num}: '{sample_quote}'")
    print("=" * 60)

    start_time = perf_counter()

    # Encode the query sentence on the appropriate device
    inputs = tokenizer(
        [sample_quote],
        padding=True,
        truncation=True,
        return_tensors="np"
    )
    input_ids = jnp.array(inputs["input_ids"])

    token_embeddings = model(input_ids)
    embedding = jnp.mean(token_embeddings, axis=1)

    # Note: don't forget to convert jax arrays to numpy arrays before passing to FAISS
    # Normalize for cosine similarity
    embedding_np = np.array(embedding, dtype=np.float32)
    faiss.normalize_L2(embedding_np)


    # Search top matches
    D, I = index.search(embedding_np, BEST_K)

    end_time = perf_counter()

    # Step through the results
    for i in range(BEST_K):
        idx = int(I[0, i])
        distance = D[0, i]
        print(f"{idx}", file=results_file)

        cur.execute("SELECT data FROM quotes WHERE id = ?", (idx,))
        row = cur.fetchone()
        if row is None:
            sentence = "<missing quote>"
            attributes = "<missing author>"
        else:
            sentence, attributes = pickle.loads(row[0])

        print(f"\n{idx},{distance:.3f}: '{sentence}'\n\t{attributes}")

    print("=====", file=results_file)
    print(f"Time taken: {end_time - start_time:.3f} seconds")
    print("\n")

results_file.close()
conn.close()
