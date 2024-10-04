from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings1 = model.encode("This is a sentence")
embeddings2 = model.encode("This is another sentence")
similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
print(similarity)
