"""Run the mteb benchmark to evaluate the model embeddings."""

import mteb

import benchmark

if __name__ == "__main__":
    mteb_model = mteb.MTEB(tasks=["Banking77Classification", "RedditClustering", "SummEval"])
    mteb_model.run(model=benchmark.FauxModel())
