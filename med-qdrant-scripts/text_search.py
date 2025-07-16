import os, sys
sys.path.append(os.path.abspath("./"))
import time
import torch
import torch.nn.functional as F
from functools import lru_cache
from pprint import pprint
import itertools

from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient

from utils import concat
from rerank import get_reranked_scores

# --- New Model and Tokenizer Initialization ---
print("Initializing Qwen3-Embedding-0.6B model and tokenizer...")
try:
    # 推荐在支持的环境中启用 flash_attention_2 以获得更好的性能
    dense_model = AutoModel.from_pretrained('/mnt/public/lianghao/wzr/med_reseacher/med-retrieval/model/Qwen3-Embedding-0.6B', attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda().eval()
    # dense_model = AutoModel.from_pretrained('/mnt/public/lianghao/wzr/med_reseacher/med-retrieval/model/Qwen3-Embedding-0.6B')
    tokenizer = AutoTokenizer.from_pretrained('/mnt/public/lianghao/wzr/med_reseacher/med-retrieval/model/Qwen3-Embedding-0.6B', padding_side='left')
    print("Qwen3-Embedding-0.6B model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading Qwen3-Embedding-0.6B model: {e}", file=sys.stderr)
    sys.exit(1)

# --- New Helper Functions from qwen3_embed_transformer.py ---

def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """
    从模型的最后一层隐藏状态中提取嵌入向量。
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    根据Qwen3-Embedding的要求，为查询构建带有指令的输入字符串。
    """
    return f'Instruct: {task_description}\nQuery:{query}'

# 定义检索任务的默认指令
DEFAULT_RETRIEVAL_TASK = 'Given a web search query, retrieve relevant passages that answer the query'
MAX_LENGTH = 512 # Qwen3-Embedding-0.6B 的最大序列长度

# --- Qdrant Client Initialization ---
client = QdrantClient(url="http://localhost:6333")

SINGLE_TEXT_TOOLS = ["guideline_qwen", "research_qwen", "book_qwen", "wiki_qwen"]
ADAPTIVE_TEXT_TOOLS = ["adaptive_text"]


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
def run_qdrant_search_with_embedding(tool, dense_embed, retrievel_topk):
    """
    [Unchanged] 使用预先计算好的查询嵌入执行Qdrant搜索。
    """
    assert tool in SINGLE_TEXT_TOOLS
    start_time = time.time()
    
    # 注意: Qdrant中的`using`参数应与集合创建时向量索引的命名一致。
    qwen3_search_res = client.query_points(
        collection_name=tool,
        query_filter=None,
        query=dense_embed.tolist(),
        using="qwen3-embedding-0.6b",
        limit=retrievel_topk,
        timeout=300
    ).points
    qwen3_search_res = extract_points(qwen3_search_res)
    end_time = time.time()
    return qwen3_search_res, end_time - start_time


@torch.no_grad()
def run_qdrant_search(tool, query, retrievel_topk):
    """
    [Updated] 使用新的Qwen3-Embedding模型生成嵌入向量并执行搜索。
    """
    assert tool in SINGLE_TEXT_TOOLS
    
    # --- 使用Qwen3-Embedding生成查询嵌入 ---
    # 1. 格式化查询
    formatted_query = get_detailed_instruct(DEFAULT_RETRIEVAL_TASK, query)
    
    # 2. 编码
    encoded = tokenizer(
        [formatted_query],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt',
    )
    encoded.to(dense_model.device) # 确保输入张量在模型所在的设备上
    
    # 3. 模型推理
    outputs = dense_model(**encoded)
    
    # 4. 池化和归一化
    embedding = last_token_pool(outputs.last_hidden_state, encoded['attention_mask'])
    dense_embed = F.normalize(embedding, p=2, dim=1)
    # ---------------

    # 调用使用嵌入向量的新函数
    return run_qdrant_search_with_embedding(
        tool=tool,
        dense_embed=dense_embed[0],
        retrievel_topk=retrievel_topk
    )


@lru_cache(maxsize=100000000)
def get_text_docs_adaptive(query, retrieval_topk, rerank_topk):
    """
    [Updated] 自适应检索模式，使用Qwen3-Embedding模型。
    """
    total_retrieval_time = 0
    all_retrieved_docs = []

    # --- 优化点：在此处仅执行一次查询嵌入 ---
    with torch.no_grad():
        formatted_query = get_detailed_instruct(DEFAULT_RETRIEVAL_TASK, query)
        encoded = tokenizer(
            [formatted_query],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt',
        )
        encoded.to(dense_model.device)
        outputs = dense_model(**encoded)
        embedding = last_token_pool(outputs.last_hidden_state, encoded['attention_mask'])
        dense_embed = F.normalize(embedding, p=2, dim=1)
    # -----------------------------------------

    # 1. 在所有文本源中并行进行向量检索，复用同一个嵌入向量
    for tool in SINGLE_TEXT_TOOLS:
        docs, retrieval_time = run_qdrant_search_with_embedding(
            tool=tool,
            dense_embed=dense_embed[0], # 复用嵌入
            retrievel_topk=retrieval_topk
        )
        all_retrieved_docs.extend(docs)
        total_retrieval_time += retrieval_time

    # 2. 对合并后的结果进行统一重排
    if not all_retrieved_docs:
        return [], total_retrieval_time, 0

    unique_docs = {doc['id']: doc for doc in all_retrieved_docs}.values()

    scores, rerank_time = get_reranked_scores(
        query=query,
        articles=[concat(i['title'], i['para']) for i in unique_docs],
    )

    for sco, res in zip(scores, unique_docs):
        res["rerank_score"] = sco
    
    # 3. 按重排分数排序并返回topk
    sorted_results = sorted(list(unique_docs), key=lambda i: i['rerank_score'], reverse=True)
    final_results = sorted_results[:rerank_topk]

    return final_results, total_retrieval_time, rerank_time


@lru_cache(maxsize=100000000)
def get_text_docs(tool, query, retrieval_topk, rerank_topk):
    # 1. vector search
    if tool in SINGLE_TEXT_TOOLS:
        # 此处调用已更新为使用新模型
        all_results, retrieval_time = run_qdrant_search(
            tool=tool,
            query=query,
            retrievel_topk=retrieval_topk
        )
    else:
        raise NotImplementedError
        
    # 2. rerank
    scores, rerank_time = get_reranked_scores(
        query=query,
        articles=[concat(i['title'], i['para']) for i in all_results],
    )
    for sco, res in zip(scores, all_results):
        res["rerank_score"] = sco
    all_results.sort(key=lambda i: i['rerank_score'], reverse=True)
    all_results = all_results[:rerank_topk]

    return all_results, retrieval_time, rerank_time
    

if __name__ == "__main__":
    # --- 测试用例 ---
    print("--- Running Standard Text Search Example with Qwen3-Embedding ---")
    results, retrieval_time, rerank_time = get_text_docs(tool="wiki", query="Prunus incisa Thunb. – Fuji cherry Prunus jamasakura Siebold ex Koidz. – Japanese mountain cherry or Japanese hill cherry Prunus leveilleana (Koidz.)", retrieval_topk=20, rerank_topk=10)
    pprint(results, width=100)
    print(f"Retrieval Time: {retrieval_time:.4f}s")
    print(f"Rerank Time: {rerank_time:.4f}s")
    print("="*89)

    print("\n--- Running Adaptive Text Search Example with Qwen3-Embedding ---")
    adaptive_query = "fenofibrate and sleep apnoea"
    results_adaptive, retrieval_time_adaptive, rerank_time_adaptive = get_text_docs_adaptive(query=adaptive_query, retrieval_topk=20, rerank_topk=10)
    pprint(results_adaptive, width=100)
    print(f"Adaptive Retrieval Time: {retrieval_time_adaptive:.4f}s")
    print(f"Adaptive Rerank Time: {rerank_time_adaptive:.4f}s")
    print("="*89)