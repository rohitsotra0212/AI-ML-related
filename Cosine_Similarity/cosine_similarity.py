from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key= os.getenv("OPENAI_API_KEY"))

s1 = "your date of joining is 12/07/2019"
s2 = "what is date of joining"

emb1 = model.encode(s1)
emb2 = model.encode(s2)
#openai_emb = embeddings.embed_query(s1)
#openai_emb = np.array(openai_emb)

print(f"emd1 \n {emb1.shape}")
#print(f"openai_emb \n {openai_emb.shape}")
#print(f"openai_emb Length \n {len(openai_emb)}")

cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

print("Cosine Similarity Score:", cos_sim)
percentage = cos_sim * 100
print(f"Cosine Similarity %: {percentage:.2f}%")

