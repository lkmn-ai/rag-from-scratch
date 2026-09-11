# End-to-End RAG System with ChromaDB, Sentence Transformers & FLAN-T5

A beginner-friendly **Retrieval-Augmented Generation (RAG)** system built in Python.

This project demonstrates how to build a complete RAG pipeline from scratch using:

* Python
* PyPDF
* LangChain Text Splitters
* Sentence Transformers
* ChromaDB
* Hugging Face Transformers
* FLAN-T5
* PyTorch
* Pandas
* NumPy

The goal of this project is to understand **how RAG works internally**, rather than hiding the complete process behind a high-level RAG framework.

---

# 1. What is RAG?

**RAG = Retrieval-Augmented Generation**

A normal Large Language Model (LLM) generates an answer based on what it learned during training.

The problem is that an LLM:

* May not know information from our private documents.
* May not have the latest information.
* Can hallucinate facts.
* Does not automatically know the contents of our local PDFs or text files.

RAG solves this by giving the LLM relevant information from our documents **before asking it to generate the answer**.

The basic idea is:

```text
User Question
      ↓
Retrieve relevant information
      ↓
Give that information to the LLM
      ↓
Generate answer using the retrieved information
```

Therefore:

```text
RAG = Retrieval + Generation
```

---

# 2. Project Architecture

This project has two major pipelines:

## Indexing Pipeline

The indexing pipeline prepares documents so that they can later be searched.

```text
PDF / TXT Documents
        ↓
    Text Extraction
        ↓
       Chunking
        ↓
     Embeddings
        ↓
    ChromaDB
```

## RAG / Query Pipeline

The RAG pipeline answers a user's question.

```text
User Question
      ↓
Question Embedding
      ↓
ChromaDB Similarity Search
      ↓
Top-K Relevant Chunks
      ↓
Build Context
      ↓
Build Prompt
      ↓
FLAN-T5
      ↓
Final Answer
```

Complete architecture:

```text
                  INDEXING PIPELINE
                  =================

             PDF / TXT Documents
                      ↓
                Extract Text
                      ↓
                  Chunk Text
                      ↓
              Embedding Model
             all-MiniLM-L6-v2
                      ↓
                  Vectors
                      ↓
                  ChromaDB
                      │
══════════════════════╪══════════════════════
                      │
                      ↓
                    QUERY
                      ↓
                User Question
                      ↓
              Embedding Model
                      ↓
                Query Vector
                      ↓
                  ChromaDB
                      ↓
                Similarity Search
                      ↓
                  Top-K Chunks
                      ↓
               build_context()
                      ↓
                    Context
                      ↓
             build_rag_prompt()
                      ↓
                    Prompt
                      ↓
                  Tokenizer
                      ↓
                   FLAN-T5
                      ↓
               Generated Tokens
                      ↓
              tokenizer.decode()
                      ↓
                Final Answer
```

---

# 3. Technologies Used

| Technology                | Purpose                             |
| ------------------------- | ----------------------------------- |
| Python                    | Main programming language           |
| PyPDF                     | Extract text from PDF files         |
| LangChain Text Splitters  | Split documents into chunks         |
| Sentence Transformers     | Convert text into embeddings        |
| `all-MiniLM-L6-v2`        | Embedding model                     |
| ChromaDB                  | Store and search embeddings         |
| Hugging Face Transformers | Load and run FLAN-T5                |
| `google/flan-t5-base`     | Generation model                    |
| PyTorch                   | Run the neural network              |
| Pandas                    | Inspect and organize retrieved data |
| NumPy                     | Work with embedding vectors         |

---

# 4. Installation

Install the required packages:

```python
%pip install -q pypdf langchain-text-splitters sentence-transformers chromadb transformers accelerate
```

Avoid unnecessarily upgrading packages such as NumPy, Pandas, or PyTorch in a managed environment like Google Colab because the environment may already contain compatible versions.

After installation, restart the runtime if required.

---

# 5. Imports and Configuration

```python
import os
import glob
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
)

import torch
import chromadb

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
```

These imports provide the tools required for:

* Reading files
* Extracting PDF text
* Splitting documents
* Creating embeddings
* Storing vectors
* Loading the generation model
* Running inference

---

# 6. Configuration

```python
DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

TOP_K = 4

DATA_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)
print("Data:", DATA_DIR.resolve())
print("Embedding model:", EMBEDDING_MODEL_NAME)
print("Generation model:", GENERATION_MODEL_NAME)
```

## Important configuration values

### `DATA_DIR`

```python
DATA_DIR = Path("./data")
```

This is where our PDF and TXT documents are stored.

### `CHROMA_DIR`

```python
CHROMA_DIR = Path("./chroma_db")
```

This is where ChromaDB stores its persistent local database.

### Embedding model

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
```

This model converts text into numerical vectors.

Its embeddings have a dimension of **384**.

### Generation model

```python
GENERATION_MODEL_NAME = "google/flan-t5-base"
```

This model generates the final answer from our prompt.

### `CHUNK_SIZE`

```python
CHUNK_SIZE = 800
```

Controls the approximate maximum size of each text chunk.

### `CHUNK_OVERLAP`

```python
CHUNK_OVERLAP = 120
```

Controls how much text is shared between neighboring chunks.

Conceptually:

```text
Chunk 1:
0 ---------------- 800

Chunk 2:
             680 ---------------- 1480

Chunk 3:
                          1360 ---------------- 2160
```

The overlap is approximately:

```text
800 - 120 = 680
```

However, `RecursiveCharacterTextSplitter` does not necessarily produce exact fixed boundaries because it tries to split at meaningful separators first.

### `TOP_K`

```python
TOP_K = 4
```

This means that by default we retrieve the **4 most relevant chunks**.

### `DEVICE`

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

If a GPU is available:

```text
DEVICE = cuda
```

Otherwise:

```text
DEVICE = cpu
```

---

# 7. Sample Documents

For testing, the project can create sample documents if no PDF or TXT files exist.

```python
sample_docs = {
    "accessibility_manual.txt": """...""",
    "account_management_manual.txt": """..."""
}

if not list(DATA_DIR.glob("*.pdf")) and not list(DATA_DIR.glob("*.txt")):
    for filename, content in sample_docs.items():
        (
            DATA_DIR / filename
        ).write_text(
            textwrap.dedent(content).strip(),
            encoding="utf-8"
        )

    print("No documents found. Created sample documents.")
else:
    print("Existing documents found; sample documents were not created.")
```

The important logic is:

```python
DATA_DIR.glob("*.pdf")
DATA_DIR.glob("*.txt")
```

These search for PDF and TXT files inside the data directory.

If no documents exist, sample documents are created.

If documents already exist, they are preserved.

---

# 8. Loading Documents

The next step is to extract text from our documents.

```python
def load_documents(data_dir: Path):
    documents = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                documents.append({
                    "doc_id": pdf_path.stem,
                    "source": pdf_path.name,
                    "page": page_number,
                    "content": text,
                })

    for txt_path in sorted(data_dir.glob("*.txt")):
        text = txt_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if text:
            documents.append({
                "doc_id": txt_path.stem,
                "source": txt_path.name,
                "page": None,
                "content": text,
            })

    return documents
```

## What does this function do?

It converts:

```text
PDF / TXT files
      ↓
Extracted text
```

The output is a Python list containing dictionaries.

For example:

```python
[
    {
        "doc_id": "accessibility_manual",
        "source": "accessibility_manual.txt",
        "page": None,
        "content": "Warranty: The device..."
    }
]
```

Each dictionary represents a document or PDF page.

---

# 9. Why Store Metadata?

Every document contains more than just text.

We also store:

```text
doc_id
source
page
content
```

For example:

```python
{
    "doc_id": "manual",
    "source": "manual.pdf",
    "page": 5,
    "content": "The warranty is..."
}
```

This allows us to later identify where the retrieved information came from.

Metadata is extremely useful for:

* Source attribution
* Citations
* Debugging
* Showing document names
* Showing PDF page numbers

---

# 10. Convert Documents to a DataFrame

```python
documents = load_documents(DATA_DIR)

documents_df = pd.DataFrame(documents)

print("Documents/pages loaded:", len(documents_df))

display(documents_df.head())
```

The original data structure is:

```text
List
 ↓
Dictionary
 ↓
Dictionary
 ↓
Dictionary
```

Pandas converts this into a table that is easier to inspect.

---

# 11. Chunking

Large documents should not normally be embedded as one giant piece of text.

Instead, we split them into smaller chunks.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

## Why chunk documents?

Suppose a document contains 50 pages.

If we create one embedding for the entire document, a question about one small section may not retrieve the most relevant information accurately.

Instead:

```text
Large document
      ↓
Small chunks
      ↓
Embedding for each chunk
      ↓
Search individual chunks
```

This makes retrieval more precise.

---

# 12. Recursive Character Text Splitter

The separators are:

```python
[
    "\n\n",
    "\n",
    ". ",
    " ",
    ""
]
```

The splitter tries to split in this order:

```text
1. Paragraph
      ↓
2. Line
      ↓
3. Sentence
      ↓
4. Word/space
      ↓
5. Individual characters
```

This helps avoid cutting text in unnatural places when possible.

---

# 13. Creating Chunks

```python
chunks = []

for doc in documents:
    for chunk_index, chunk in enumerate(
        splitter.split_text(doc["content"])
    ):
        chunks.append({
            "chunk_id": f'{doc["doc_id"]}_{doc["page"]}_{chunk_index}',
            "doc_id": doc["doc_id"],
            "source": doc["source"],
            "page": doc["page"],
            "chunk_index": chunk_index,
            "content": chunk,
        })
```

Each chunk receives:

```text
chunk_id
doc_id
source
page
chunk_index
content
```

For example:

```text
manual_None_0
manual_None_1
manual_None_2
```

The `chunk_id` uniquely identifies a chunk.

---

# 14. Chunk DataFrame

```python
chunks_df = pd.DataFrame(chunks)

print("Total chunks:", len(chunks_df))

display(chunks_df.head())
```

Now the document has been transformed into:

```text
Documents
   ↓
Chunks
   ↓
DataFrame
```

---

# 15. Inspecting Chunks

```python
for i, row in chunks_df.head(5).iterrows():
    print(
        f"--- Chunk {i} | "
        f"{row['source']} | "
        f"page={row['page']} ---"
    )

    print(row["content"][:700])
    print()
```

This is mainly for debugging.

It allows us to verify:

* Are chunks being created correctly?
* Is important text preserved?
* Is the chunk size reasonable?
* Is metadata correct?

---

# 16. Embeddings

Now we convert every chunk into a vector.

First load the embedding model:

```python
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=DEVICE
)
```

The embedding model is:

```text
all-MiniLM-L6-v2
```

---

# 17. What is an Embedding?

An embedding converts text into a numerical representation.

For example:

```text
"How long is the warranty?"
            ↓
        Embedding Model
            ↓
[0.12, -0.45, 0.78, ...]
```

The vector contains many numerical values.

For `all-MiniLM-L6-v2`, the embedding dimension is typically:

```text
384
```

The important idea is:

```text
Similar meaning
      ↓
Similar vectors
```

Therefore embeddings allow us to perform semantic search.

---

# 18. Extract Chunk Text

```python
chunk_texts = chunks_df["content"].tolist()
```

We only need the text for embedding.

This converts:

```text
DataFrame column
      ↓
Python list
```

For example:

```python
[
    "Warranty information...",
    "Accessibility guidelines...",
    "Account management..."
]
```

---

# 19. Create Embeddings

```python
chunk_embeddings = embedding_model.encode(
    chunk_texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)
```

This converts every chunk into a vector.

Conceptually:

```text
Chunk 1 → Vector 1
Chunk 2 → Vector 2
Chunk 3 → Vector 3
...
```

### `batch_size=32`

Processes up to 32 chunks at a time.

### `show_progress_bar=True`

Displays progress.

### `normalize_embeddings=True`

Normalizes the embedding vectors.

This is useful when using cosine-style similarity/distance.

---

# 20. Embedding Matrix

```python
chunk_embeddings = np.asarray(chunk_embeddings)

print("Embedding matrix shape:", chunk_embeddings.shape)

print(
    "Embedding dimension:",
    chunk_embeddings.shape[1]
)
```

If there are 100 chunks:

```text
Embedding matrix shape:

(100, 384)
```

This means:

```text
100 → number of chunks
384 → vector dimensions
```

So:

```text
100 chunks
   ↓
100 vectors
   ↓
each vector has 384 numbers
```

---

# 21. ChromaDB

Now we need somewhere to store:

* Chunk IDs
* Chunk text
* Embeddings
* Metadata

We use ChromaDB.

```python
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)
```

`PersistentClient` means the database is stored locally rather than existing only in memory.

---

# 22. Create a Collection

```python
collection_name = "rag_training_demo"
```

A ChromaDB collection is similar to a database table containing our vectors and associated information.

We remove the old collection:

```python
try:
    chroma_client.delete_collection(collection_name)
except Exception:
    pass
```

This is useful when repeatedly running the notebook.

Otherwise old data may remain and cause duplicate IDs or stale results.

Then create a new collection:

```python
collection = chroma_client.create_collection(
    name=collection_name,
    metadata={
        "description":
        "Open-source RAG training collection"
    }
)
```

---

# 23. Store Data in ChromaDB

```python
collection.add(
    ids=chunks_df["chunk_id"].tolist(),

    documents=chunks_df["content"].tolist(),

    embeddings=chunk_embeddings.tolist(),

    metadatas=chunks_df[
        ["doc_id", "source", "page", "chunk_index"]
    ]
    .fillna("")
    .to_dict(orient="records")
)
```

Each ChromaDB record conceptually contains:

```text
ID
 +
Text
 +
Embedding
 +
Metadata
```

Example:

```text
ID:
manual_None_0

Text:
"Warranty: The device..."

Embedding:
[0.12, -0.45, ...]

Metadata:
{
    "source": "manual.txt",
    "page": "",
    "chunk_index": 0
}
```

---

# 24. Why `.fillna("")`?

Some values such as PDF page numbers can be missing.

For TXT documents:

```text
page = None
```

Pandas may represent this as:

```text
NaN
```

So:

```python
.fillna("")
```

replaces missing values with an empty string.

---

# 25. Why `to_dict(orient="records")`?

The metadata needs to be supplied as a list of dictionaries.

For example:

```python
[
    {
        "doc_id": "manual",
        "source": "manual.txt",
        "page": "",
        "chunk_index": 0
    },
    {
        "doc_id": "manual",
        "source": "manual.txt",
        "page": "",
        "chunk_index": 1
    }
]
```

`orient="records"` creates exactly this structure.

---

# 26. Retrieval

Now we move to the query side.

The user asks a question:

```text
"How long is the warranty?"
```

We need to find the most relevant chunks.

```python
def retrieve(
    question: str,
    top_k: int = TOP_K
):
```

---

# 27. Convert Question to Embedding

```python
query_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True
)[0]
```

The same embedding model used for documents must be used for the question.

This is important.

```text
Document chunks
      ↓
Embedding Model
      ↓
Document vectors
```

and:

```text
User question
      ↓
Same Embedding Model
      ↓
Question vector
```

Both are therefore represented in the same vector space.

---

# 28. Why `[question]`?

The embedding model expects a collection/list of texts.

Instead of:

```python
question
```

we pass:

```python
[question]
```

which is a list containing one question.

The result is therefore a batch containing one embedding.

Then:

```python
[0]
```

gets the first embedding.

---

# 29. Similarity Search in ChromaDB

```python
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=top_k,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)
```

This tells ChromaDB:

> Find the `top_k` chunks whose vectors are closest to the question vector.

---

# 30. What is Distance?

ChromaDB calculates the distance between:

```text
Question vector
       ↓
Chunk 1 vector
Chunk 2 vector
Chunk 3 vector
...
```

For example:

```text
Chunk 1 → 0.834513
Chunk 2 → 1.813772
Chunk 3 → 2.004651
```

For the distance metric being used here:

```text
Smaller distance
       ↓
More similar
```

Therefore:

```text
0.834513 → best match
1.813772 → second
2.004651 → third
```

We do not manually calculate this distance in our Python code.

ChromaDB performs the vector comparison and returns the distances.

---

# 31. Why Are ChromaDB Results Nested?

For one query, ChromaDB returns something conceptually like:

```python
results["documents"] = [
    [
        "chunk 1",
        "chunk 2",
        "chunk 3",
        "chunk 4"
    ]
]
```

The outer list represents the query.

The inner list contains the retrieved chunks.

Therefore:

```python
results["documents"][0]
```

means:

> Get the first query's list of retrieved documents.

And:

```python
results["documents"][0][i]
```

means:

> Get the `i`th retrieved chunk from the first query.

Example:

```python
documents = [
    ["chunk 1", "chunk 2", "chunk 3"]
]
```

Then:

```python
documents[0]
```

returns:

```python
["chunk 1", "chunk 2", "chunk 3"]
```

while:

```python
documents[0][0]
```

returns:

```text
"chunk 1"
```

---

# 32. Convert Retrieval Results into a DataFrame

```python
rows = []

for i in range(
    len(results["documents"][0])
):
    rows.append({
        "rank": i + 1,

        "content":
            results["documents"][0][i],

        "metadata":
            results["metadatas"][0][i],

        "distance":
            results["distances"][0][i],
    })
```

This reorganizes ChromaDB's nested response into a simple table.

Each retrieved result becomes one dictionary:

```python
{
    "rank": 1,
    "content": "...",
    "metadata": {...},
    "distance": 0.83
}
```

The important thing is that the indexes line up:

```text
documents[0][i]  → chunk text

metadatas[0][i]  → metadata for that chunk

distances[0][i]  → distance for that chunk
```

The same `i` refers to the same retrieved result.

---

# 33. Example Retrieval Result

For:

```text
How long is the warranty?
```

we might get:

```text
rank | distance | source
--------------------------------
1    | 0.834513 | accessibility_manual.txt
2    | 1.813772 | accessibility_manual.txt
3    | 2.004651 | account_management_manual.txt
```

The first chunk is the most relevant because it has the smallest distance.

---

# 34. Building Context

Once relevant chunks are retrieved, we combine them into a single string.

```python
def build_context(
    retrieved_df: pd.DataFrame
) -> str:

    context_parts = []

    for _, row in retrieved_df.iterrows():

        metadata = row["metadata"]

        page_info = (
            f", page {metadata['page']}"
            if metadata.get("page") not in ("", None)
            else ""
        )

        context_parts.append(
            f"[Source: {metadata['source']}{page_info}]\n"
            f"{row['content']}"
        )

    return "\n\n".join(context_parts)
```

---

# 35. What Does `_` Mean?

This line:

```python
for _, row in retrieved_df.iterrows():
```

could also be written:

```python
for index, row in retrieved_df.iterrows():
```

`iterrows()` gives:

```text
(index, row)
```

We don't need the index, so `_` is used as a convention meaning:

```text
"I don't need this value."
```

Therefore:

```text
_    → ignore index
row  → use the actual row
```

---

# 36. Context Format

If three chunks are retrieved, the context might become:

```text
[Source: accessibility_manual.txt]
Warranty:
The device is covered by a 24-month warranty.

[Source: accessibility_manual.txt]
Accessibility Support Guidelines
...

[Source: account_management_manual.txt]
Customer Account Management Guide
...
```

This is the information that will be provided to the generation model.

---

# 37. Building the RAG Prompt

```python
def build_rag_prompt(
    question: str,
    context: str
) -> str:

    return f"""
Answer the question using ONLY the context provided below.

Rules:
- Do not invent facts.
- If the answer is not present in the context, say:
  "I could not find this information in the provided documents."
- Keep the answer concise and factual.
- Mention the relevant source when possible.

Context:
{context}

Question:
{question}

Answer:
""".strip()
```

This function combines:

```text
Instructions
     +
Retrieved context
     +
User question
     ↓
Final prompt
```

---

# 38. Why Give Instructions to the LLM?

The model needs to know how it should use the retrieved information.

For example:

```text
Answer using ONLY the context.
```

helps reduce hallucination.

This instruction:

```text
Do not invent facts.
```

tells the model not to make up information.

And:

```text
If the answer is not present...
```

provides a fallback response when the retrieved context does not contain the answer.

---

# 39. Example Final Prompt

The final prompt may look like:

```text
Answer the question using ONLY the context provided below.

Rules:
- Do not invent facts.
- If the answer is not present in the context, say:
  "I could not find this information in the provided documents."
- Keep the answer concise and factual.
- Mention the relevant source when possible.

Context:

[Source: accessibility_manual.txt]
Warranty:
The device is covered by a 24-month warranty.

Question:
How long is the warranty?

Answer:
```

This is what we give to FLAN-T5.

---

# 40. Loading FLAN-T5

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
```

We need two components:

```text
Tokenizer
+
Generation Model
```

---

# 41. Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL_NAME
)
```

The tokenizer converts human-readable text into tokens/token IDs that the model can process.

Conceptually:

```text
"How long is the warranty?"
            ↓
        Tokenizer
            ↓
      Token IDs
```

The exact token IDs depend on the tokenizer.

---

# 42. Generation Model

```python
model = AutoModelForSeq2SeqLM.from_pretrained(
    GENERATION_MODEL_NAME
)
```

This loads the actual FLAN-T5 model weights.

The tokenizer understands how to represent the text.

The model performs the actual generation.

---

# 43. Move Model to Device

```python
model = model.to(DEVICE)
```

If:

```text
DEVICE = "cuda"
```

the model is moved to the GPU.

If:

```text
DEVICE = "cpu"
```

the model runs on the CPU.

The model and its input tensors need to be on the same device.

---

# 44. Generating the Answer

```python
def generate_answer(
    prompt,
    max_new_tokens=200
):
```

This function takes the final RAG prompt and sends it to FLAN-T5.

---

# 45. Tokenizing the Prompt

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)
```

The prompt is converted into tensors.

The resulting `inputs` typically contains:

```python
{
    "input_ids": ...,
    "attention_mask": ...
}
```

---

# 46. What is `input_ids`?

`input_ids` are the numerical token IDs representing the prompt.

Conceptually:

```text
Prompt
  ↓
Tokenizer
  ↓
input_ids
  ↓
[101, 205, 300, ...]
```

The model processes these token IDs.

---

# 47. What is `attention_mask`?

The attention mask tells the model which token positions are valid and which positions are padding.

For example:

```text
Tokens:
[How, long, is, warranty, ?, PAD, PAD]

Attention mask:
[ 1,    1,   1,    1,     1,  0,   0]
```

Meaning:

```text
1 → real token
0 → padding
```

The attention mask is **not a semantic importance score**.

It primarily identifies valid token positions versus padding.

---

# 48. Move Inputs to the Device

```python
inputs = {
    k: v.to(DEVICE)
    for k, v in inputs.items()
}
```

The tokenizer gives us multiple tensors, such as:

```text
input_ids
attention_mask
```

This line moves all of them to the same device as the model.

It is essentially equivalent to:

```python
inputs["input_ids"] = (
    inputs["input_ids"].to(DEVICE)
)

inputs["attention_mask"] = (
    inputs["attention_mask"].to(DEVICE)
)
```

The shorter version is called a **dictionary comprehension**.

---

# 49. Disable Gradients

```python
with torch.no_grad():
```

We are doing inference, not training.

During training, the model needs gradients to update its weights.

During inference:

```text
Input
 ↓
Model
 ↓
Prediction
```

No weight updates are required.

`torch.no_grad()` therefore reduces unnecessary computation and memory usage.

---

# 50. Generate Tokens

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=0.2,
    do_sample=False
)
```

This is where FLAN-T5 actually generates the answer.

The flow is:

```text
input_ids
+
attention_mask
       ↓
    FLAN-T5
       ↓
Generated token IDs
```

---

# 51. What is `**inputs`?

Suppose:

```python
inputs = {
    "input_ids": input_tensor,
    "attention_mask": attention_tensor
}
```

Then:

```python
model.generate(**inputs)
```

unpacks the dictionary.

Conceptually, it becomes:

```python
model.generate(
    input_ids=input_tensor,
    attention_mask=attention_tensor
)
```

---

# 52. `max_new_tokens`

```python
max_new_tokens=200
```

This controls the maximum number of new tokens the model can generate for the answer.

It does not mean that exactly 200 tokens will always be generated.

It means:

```text
Maximum generated tokens = 200
```

---

# 53. `do_sample=False`

```python
do_sample=False
```

means the generation is deterministic rather than randomly sampling from possible next tokens.

This is useful for a factual RAG system where we want consistent answers.

---

# 54. Temperature

```python
temperature=0.2
```

Temperature controls randomness when sampling is used.

However, because:

```python
do_sample=False
```

sampling is disabled, so temperature does not meaningfully affect this particular decoding setup.

---

# 55. Decode the Output

The model generates token IDs, not normal text.

```python
return tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
```

`outputs` is typically batched.

Conceptually:

```python
outputs = [
    [123, 456, 789, 321]
]
```

Therefore:

```python
outputs[0]
```

gets the generated token IDs for the first input.

Then:

```python
tokenizer.decode(...)
```

converts those token IDs back into human-readable text.

So:

```text
Generated token IDs
       ↓
tokenizer.decode()
       ↓
Normal text
```

---

# 56. `skip_special_tokens=True`

Models can use special internal tokens.

For example:

```text
<EOS>
<PAD>
```

We normally don't want these in the final answer.

Therefore:

```python
skip_special_tokens=True
```

removes them from the decoded output.

---

# 57. Complete RAG Function

Now all the individual functions can be combined.

```python
def rag(
    question: str,
    top_k: int = TOP_K
):

    retrieved = retrieve(
        question,
        top_k=top_k
    )

    context = build_context(
        retrieved
    )

    prompt = build_rag_prompt(
        question,
        context
    )

    answer = generate_answer(
        prompt
    )

    return {
        "question": question,
        "answer": answer,
        "retrieved": retrieved,
        "context": context,
        "prompt": prompt,
    }
```

This is the **orchestrator** of the entire RAG system.

It calls each stage in the correct order.

---

# 58. Calling the RAG System

```python
result = rag(
    "What are the recommended performance optimization practices?"
)
```

One function call now executes the entire pipeline:

```text
Question
   ↓
retrieve()
   ↓
Relevant chunks
   ↓
build_context()
   ↓
Context
   ↓
build_rag_prompt()
   ↓
Prompt
   ↓
generate_answer()
   ↓
FLAN-T5
   ↓
Final Answer
```

---

# 59. What Does `result` Contain?

The `rag()` function returns a dictionary:

```python
{
    "question": question,
    "answer": answer,
    "retrieved": retrieved,
    "context": context,
    "prompt": prompt,
}
```

Therefore:

```python
result["question"]
```

returns the original question.

```python
result["answer"]
```

returns the generated answer.

```python
result["retrieved"]
```

returns the retrieved chunks DataFrame.

```python
result["context"]
```

returns the combined context.

```python
result["prompt"]
```

returns the final prompt sent to the generation model.

This is very useful for debugging because we can inspect every stage of the RAG pipeline.

---

# 60. Displaying the Result

```python
print("QUESTION:")
print(result["question"])

print("\nANSWER:")
print(result["answer"])

print("\nRETRIEVED SOURCES:")

display(
    result["retrieved"][
        ["rank", "distance", "metadata"]
    ]
)
```

This displays:

```text
QUESTION:
What are the recommended performance optimization practices?

ANSWER:
...

RETRIEVED SOURCES:
rank | distance | metadata
...
```

---

# 61. End-to-End Data Flow

The most important thing to understand from this project is how information moves through the system.

## Step 1 — Documents

```text
PDF / TXT
```

↓

## Step 2 — Text Extraction

```text
Raw files
    ↓
Extracted text
```

↓

## Step 3 — Chunking

```text
Large text
    ↓
Small chunks
```

↓

## Step 4 — Embeddings

```text
Chunk
    ↓
Embedding model
    ↓
Vector
```

↓

## Step 5 — Vector Database

```text
Vector + Text + Metadata
          ↓
       ChromaDB
```

↓

## Step 6 — User Query

```text
"What is the warranty?"
```

↓

## Step 7 — Query Embedding

```text
Question
    ↓
Embedding model
    ↓
Question vector
```

↓

## Step 8 — Similarity Search

```text
Question vector
      ↓
Compare with stored vectors
      ↓
Calculate distances
      ↓
Retrieve closest chunks
```

↓

## Step 9 — Context Building

```text
Retrieved chunks
      ↓
One context string
```

↓

## Step 10 — Prompt Construction

```text
Instructions
+
Context
+
Question
      ↓
Prompt
```

↓

## Step 11 — Tokenization

```text
Prompt
   ↓
Tokenizer
   ↓
input_ids
+
attention_mask
```

↓

## Step 12 — Generation

```text
Tokens
   ↓
FLAN-T5
   ↓
Generated token IDs
```

↓

## Step 13 — Decoding

```text
Generated token IDs
       ↓
tokenizer.decode()
       ↓
Human-readable answer
```

---

# 62. Why Do We Need Two Models?

A common confusion in RAG is thinking that one model performs everything.

This project uses two different models.

## Embedding Model

```text
all-MiniLM-L6-v2
```

Purpose:

```text
Text → Vector
```

It is used for **retrieval**.

It helps answer:

> "Which document chunks are semantically similar to this question?"

---

## Generation Model

```text
google/flan-t5-base
```

Purpose:

```text
Prompt → Answer
```

It is used for **generation**.

It answers:

> "Given this question and retrieved context, what should I say?"

---

# 63. Embedding Model vs Generation Model

| Component | Embedding Model       | Generation Model               |
| --------- | --------------------- | ------------------------------ |
| Model     | all-MiniLM-L6-v2      | FLAN-T5 Base                   |
| Input     | Text                  | Prompt                         |
| Output    | Vector                | Text                           |
| Purpose   | Retrieval             | Answer generation              |
| Used by   | ChromaDB search       | Final response                 |
| Example   | `"warranty"` → vector | Prompt → `"24-month warranty"` |

---

# 64. Why RAG Instead of Directly Asking the LLM?

Without RAG:

```text
Question
   ↓
LLM
   ↓
Answer
```

The model may not know our private documents.

With RAG:

```text
Question
   ↓
Retrieve relevant documents
   ↓
Give documents to LLM
   ↓
Answer
```

This allows the model to answer using information from our own document collection.

---

# 65. RAG Does Not Train the LLM

This is very important.

When we add documents to ChromaDB, we are **not training FLAN-T5**.

We are simply storing information externally.

```text
Documents
   ↓
Embeddings
   ↓
ChromaDB
```

FLAN-T5's weights remain unchanged.

During a question:

```text
Question
   ↓
Retrieve relevant chunks
   ↓
Put chunks into prompt
   ↓
FLAN-T5 reads them
   ↓
Generates answer
```

Therefore:

```text
RAG ≠ Fine-tuning
```

---

# 66. RAG vs Fine-Tuning

### RAG

Adds external knowledge at inference time.

```text
Documents → Vector DB → Prompt → LLM
```

### Fine-Tuning

Changes the model's learned weights through training.

```text
Training data
     ↓
Model training
     ↓
Updated model weights
```

RAG is especially useful when the information changes frequently or comes from private documents.

---

# 67. Important Concepts Learned

This project demonstrates several important ML/LLM concepts.

### 1. Document ingestion

Reading documents and extracting text.

### 2. Chunking

Breaking large documents into smaller searchable units.

### 3. Embeddings

Representing text as numerical vectors.

### 4. Vector databases

Storing and searching vectors efficiently.

### 5. Semantic search

Finding information based on meaning rather than exact keyword matching.

### 6. Similarity/distance

Measuring how close the query vector is to document vectors.

### 7. Retrieval

Selecting the most relevant document chunks.

### 8. Context construction

Combining retrieved chunks into a single context.

### 9. Prompt engineering

Giving the generation model clear instructions.

### 10. Tokenization

Converting text into model-readable token IDs.

### 11. Attention masks

Identifying valid token positions versus padding.

### 12. Sequence-to-sequence generation

Using FLAN-T5 to convert a prompt into an answer.

### 13. Inference

Running a trained model without updating its weights.

---

# 68. Final Mental Model

If you remember only one thing from this project, remember this:

```text
                 INDEXING

Documents
    ↓
Split into chunks
    ↓
Embedding model
    ↓
Vectors
    ↓
ChromaDB


                 QUERY

User Question
    ↓
Embedding model
    ↓
Question Vector
    ↓
ChromaDB
    ↓
Similarity Search
    ↓
Top-K Relevant Chunks
    ↓
Context
    ↓
Prompt
    ↓
Tokenizer
    ↓
input_ids + attention_mask
    ↓
FLAN-T5
    ↓
Generated Tokens
    ↓
Decode
    ↓
FINAL ANSWER
```

The key idea is:

> **RAG first retrieves the relevant knowledge, then gives that knowledge to the LLM so the LLM can generate a grounded answer.**

---

# 69. Project Outcome

By completing this project, we have implemented a basic end-to-end RAG system without relying on a high-level RAG framework.

The system can:

* Read PDF and TXT documents
* Extract document text
* Split documents into chunks
* Generate embeddings
* Store embeddings in ChromaDB
* Convert user questions into embeddings
* Perform vector similarity search
* Retrieve the top-K relevant chunks
* Build context from retrieved chunks
* Construct a grounded prompt
* Tokenize the prompt
* Run FLAN-T5
* Decode the generated tokens
* Return the final answer
* Display retrieved sources for debugging and transparency

---

# 70. Simplified One-Line Explanation

The entire project can be summarized as:

```text
Documents → Chunks → Embeddings → ChromaDB → Retrieve → Context → Prompt → FLAN-T5 → Answer
```

Or:

> **Store document knowledge as vectors, retrieve the most relevant chunks for a user's question, and provide those chunks to an LLM so it can generate a grounded answer.**

---

# 71. Future Improvements

Possible improvements to this project include:

* Better chunking strategies
* Token-based chunking
* Metadata filtering
* Hybrid search
* Reranking retrieved chunks
* Better embedding models
* Larger generation models
* Streaming responses
* Conversation memory
* Source citations
* Evaluation of retrieval quality
* Evaluation of answer quality
* Retrieval metrics such as Recall@K
* RAG evaluation frameworks
* Web-based UI
* API deployment
* FastAPI backend
* Docker deployment
* Production vector databases
* Query rewriting
* Multi-query retrieval
* Parent-document retrieval
* Cross-encoder reranking

---

# 72. Final Architecture Summary

```text
                         RAG SYSTEM
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
        INDEXING                        QUERY
             │                             │
       PDF / TXT                       Question
             │                             │
       Text Extraction               Query Embedding
             │                             │
         Chunking                         │
             │                             │
       Chunk Embeddings                    │
             │                             │
             └───────────┐     ┌───────────┘
                         ▼     ▼
                       ChromaDB
                          │
                    Similarity Search
                          │
                       Top-K Chunks
                          │
                     Build Context
                          │
                    Build Prompt
                          │
                      Tokenizer
                          │
                     FLAN-T5
                          │
                    Decode Output
                          │
                     Final Answer
```

This is the complete flow of the RAG system implemented in this project.
