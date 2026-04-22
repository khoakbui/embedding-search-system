import csv
import gzip
import sqlite3
import faiss
import pickle
from time import perf_counter
import numpy as np
import jax.numpy as jnp
import jax
import os
from helper import load_model
from transformers import AutoTokenizer
import tqdm


DB_PATH = "quotes.db"
INDEX_PATH = "quotes.index"
DIMENSION = 768
BATCH_SIZE = 1024

print("Available devices:", jax.devices())   # Should list CudaDevice if GPU is available
device = jax.devices()[0]
#device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices()[0]

# Remove index file if necessary
if os.path.isfile(INDEX_PATH):
    print(f"Removing {INDEX_PATH}")
    os.remove(INDEX_PATH)

if os.path.isfile(DB_PATH):
    print(f"Removing {DB_PATH}")
    os.remove(DB_PATH)

start_time = perf_counter()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(
    "google/embeddinggemma-300m",
    token=os.environ["HF_TOKEN"]
)
model = load_model()

print("Initializing key-value store...")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE quotes (
        id INTEGER PRIMARY KEY,
        data BLOB NOT NULL
    )
""")
conn.commit()

print("Initializing vector database...")
index = faiss.IndexFlatIP(DIMENSION)

# Embed the sentences in DIMENSION vector space using the LLM
# Normalize those so inner product is cosine similarity
# Add the vectors to a FAISS index
# Store the actual quote and its attributes in the key-value store
def process_sentences(sentences, attributes, start_index):
    start_processed = perf_counter()
    sentence_count = len(sentences)
    print(f"{start_index:,d}-{start_index + sentence_count - 1:,d}", end="", flush=True)

    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="np"
    )

    input_ids = jnp.array(inputs["input_ids"])

    # model returns token embeddings of shape (batch, seq_len, hidden_size)
    token_embeddings = model(input_ids)

    # IMPORTANT: simple mean pooling only
    sentence_embeddings = jnp.mean(token_embeddings, axis=1)

    # FAISS needs NumPy float32
    embeddings_np = np.array(sentence_embeddings, dtype=np.float32)

    # Normalize so inner product acts like cosine similarity
    faiss.normalize_L2(embeddings_np)

    # Add to FAISS
    index.add(embeddings_np)

    # Store quote and author by integer index
    rows_to_insert = []
    for i, (quote, author) in enumerate(zip(sentences, attributes)):
        idx = start_index + i
        blob = pickle.dumps((quote, author))
        rows_to_insert.append((idx, blob))

    cur.executemany("INSERT INTO quotes (id, data) VALUES (?, ?)", rows_to_insert)
    conn.commit()

    end_process = perf_counter()
    print(f":{end_process - start_processed:.1f}s ", end="", flush=True)
    # Put in an occasional newline
    if (start_index + sentence_count) % (4 * BATCH_SIZE) == 0:
        print("")

# Read the CSV file of quotes
print("Reading data, embedding, and storing...")

count = 0
with gzip.open("quotes.csv.gz", mode="rt", encoding="utf-8") as file:
    quote_reader = csv.reader(file)
    # Skip header row
    next(quote_reader, None)

    sentences = []
    attributes = []
    for row in tqdm.tqdm(quote_reader):
        sentences.append(row[0])  # quote
        attributes.append(row[1])  # author

        if len(sentences) == BATCH_SIZE:
            process_sentences(sentences, attributes, count)
            count += len(sentences)
            sentences = []
            attributes = []

    if len(sentences) > 0:
        process_sentences(sentences, attributes, count)
        count += len(sentences)

print("\nWriting vector database and closing key-value store")

# Save the index (with the vectors) to a file
faiss.write_index(index, INDEX_PATH)
conn.close()
print(f"Wrote {count} quotes to {DB_PATH} and {INDEX_PATH}")

end_time = perf_counter()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")
