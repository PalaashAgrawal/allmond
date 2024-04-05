from mteb import MTEB
from sentence_transformers import SentenceTransformer

model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)

evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/{model_name}")