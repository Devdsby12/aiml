from fastapi import FastAPI
from pydantic import BaseModel, Field

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

import torch

# =========================
# Load Embedding Model
# =========================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Load Notes
# =========================

with open("docs/notes.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]
print(f"Loaded {len(documents)} notes")

document_embeddings = embedding_model.encode(
    documents,
    convert_to_tensor=True
)

# =========================
# Load LLM
# =========================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("LLM loaded!")

# =========================
# FastAPI
# =========================

app = FastAPI()

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=100
    )

# =========================
# Retrieval
# =========================

def retrieve_context(query: str) -> str:

    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True
    )

    scores = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0),
        document_embeddings
    )

    best_index = scores.argmax().item()

    return documents[best_index]

# =========================
# LLM Generation
# =========================

def generate_response(prompt: str) -> str:

    context = retrieve_context(prompt)

    final_prompt = f"""
Context:
{context}

Question:
{prompt}

Answer the question using the context if relevant.
"""

    messages = [
        {
            "role": "user",
            "content": final_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response

# =========================
# API Route
# =========================

@app.post("/chat")
def chat(data: ChatRequest):

    answer = generate_response(
        data.message
    )

    return {
        "answer": answer
    }

# =========================
# Terminal Mode
# =========================

if __name__ == "__main__":

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        answer = generate_response(
            question
        )

        print("\nAssistant:")
        print(answer)
        print("-" * 50)
