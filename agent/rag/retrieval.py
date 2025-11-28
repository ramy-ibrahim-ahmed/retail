import os
from rank_bm25 import BM25Okapi


class LocalRetriever:
    def __init__(self, docs_dir="docs"):
        self.chunks = []
        self.doc_ids = []
        self.load_docs(docs_dir)
        tokenized_corpus = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def load_docs(self, docs_dir):
        for filename in os.listdir(docs_dir):
            if not filename.endswith(".md"):
                continue
            path = os.path.join(docs_dir, filename)
            with open(path, "r") as f:
                content = f.read()
                parts = content.split("\n\n")
                for i, part in enumerate(parts):
                    if part.strip():
                        self.chunks.append(part.strip())
                        self.doc_ids.append(f"{filename}::chunk{i}")

    def search(self, query, k=3):
        tokenized_query = query.lower().split()
        results = self.bm25.get_top_n(tokenized_query, self.chunks, n=k)
        final_res = []
        for res in results:
            idx = self.chunks.index(res)
            final_res.append({"content": res, "id": self.doc_ids[idx]})
        return final_res
