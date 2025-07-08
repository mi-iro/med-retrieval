import os, sys
sys.path.append(os.path.abspath("./"))
import time
import torch
from functools import lru_cache
from pprint import pprint

from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient

from utils import concat
from rerank import get_reranked_scores

client = QdrantClient(url="http://localhost:6333")

dense_model = AutoModel.from_pretrained("model/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("model/MedCPT-Query-Encoder")

SINGLE_TEXT_TOOLS = ["guideline", "research", "book", "wiki"]

def extract_points(points):
    results = []
    for i in points:
        results.append({
            "id": i.id,
            "title": i.payload["title"],
            "para": i.payload["para"],
            "dataset": i.payload["dataset"],
            "retrieval_score": i.score
        })
    return results


@torch.no_grad()
def run_qdrant_search(tool, query, retrievel_topk):
    assert tool in SINGLE_TEXT_TOOLS
    t1 = time.time()
    encoded = tokenizer(
        [query], 
        truncation=True, 
        padding=True, 
        return_tensors='pt', 
        max_length=64, # https://github.com/ncbi/MedCPT/issues/3#issuecomment-1874776320
    )
    # encode the queries (use the [CLS] last hidden states as the representations)
    dense_embed = dense_model(**encoded).last_hidden_state[:, 0, :]
    # sparse_embed = next(sparse_model.embed([query]))
    # print('t1', time.time() - t1)

    tmp_t = time.time()
    medcpt_search_res = client.query_points(
        collection_name=tool,
        query_filter=None,
        query=dense_embed[0].tolist(),
        using="medcpt-article",
        limit=retrievel_topk,
        timeout=300
    ).points
    medcpt_search_res = extract_points(medcpt_search_res)
    print('qdrant_search:', time.time() - tmp_t)
    return medcpt_search_res

@lru_cache(maxsize=100000000)
def get_text_docs(tool, query, retrieval_topk, rerank_topk):
    # 1. vector search
    if tool in SINGLE_TEXT_TOOLS:
        all_results = run_qdrant_search(
            tool=tool,
            query=query,
            retrievel_topk=retrieval_topk
        )
    else:
        raise NotImplementedError
    # 2. rerank
    scores = get_reranked_scores(
        query=query,
        articles=[concat(i['title'], i['para']) for i in all_results],
    )
    for sco, res in zip(scores, all_results):
        res["rerank_score"] = sco
    all_results.sort(key=lambda i: i['rerank_score'], reverse=True)
    all_results = all_results[:rerank_topk]

    return all_results
    

if __name__ == "__main__":
    pprint(get_text_docs(tool="wiki", query="Prunus incisa Thunb. – Fuji cherry Prunus jamasakura Siebold ex Koidz. – Japanese mountain cherry or Japanese hill cherry Prunus leveilleana (Koidz.)", topk=20), width=100)
    print("="*89)
