import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, sys
from tqdm import tqdm
import time

# --- [新] 全局变量和模型设置 ---

# 1. 更新模型路径
RERANKER_PATH = "/mnt/public/lianghao/wzr/med_reseacher/med-retrieval/model/Qwen3-Reranker-4B"

# 2. 设备和性能配置
# 自动检测设备 (CUDA or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# 加载 Tokenizer
# 对于 Causal LM，填充在左侧是标准做法
tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, padding_side='left')

# 加载模型并应用性能优化
# 使用 bfloat16/float16 和 flash_attention_2 以获得最佳性能
model = AutoModelForCausalLM.from_pretrained(
    RERANKER_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    attn_implementation="flash_attention_2"
).to(device).eval()

# 3. 定义模型所需的特殊 Token 和模板
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 1024*32 # Qwen3 模型支持更长的序列

# 这是 Qwen3 Reranker 的标准模板
prefix_str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix_str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix_str, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix_str, add_special_tokens=False)

# 默认指令任务
DEFAULT_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'

# --- [新] 辅助函数 ---

def format_instruction(query, doc, instruction=DEFAULT_INSTRUCTION):
    """
    根据Qwen3-Reranker的格式要求构建输入字符串。
    """
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

@torch.no_grad()
def compute_scores(batch_pairs):
    """
    处理一批(query, document)对，并计算它们的相关性分数。
    """
    # 1. 准备模型输入
    inputs = tokenizer(
        batch_pairs, 
        padding=False,  # 先不填充，手动处理
        truncation='longest_first',
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )

    # 2. 添加特殊的前后缀 Token
    for i in range(len(inputs['input_ids'])):
        inputs['input_ids'][i] = prefix_tokens + inputs['input_ids'][i] + suffix_tokens
    
    # 3. 动态填充并转换为Tensor
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    
    # 4. 将数据移至目标设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 5. 模型推理和分数计算
    logits = model(**inputs).logits
    
    # 我们只关心最后一个 token 的 logits，因为它代表了模型的预测
    last_token_logits = logits[:, -1, :]
    
    # 提取 "yes" 和 "no" 对应的 logits
    true_scores = last_token_logits[:, token_true_id]
    false_scores = last_token_logits[:, token_false_id]
    
    # 使用 LogSoftmax 将 logits 转换为概率
    scores_stack = torch.stack([false_scores, true_scores], dim=1)
    scores_log_softmax = torch.nn.functional.log_softmax(scores_stack, dim=1)
    
    # 取出 "yes" 的概率作为最终分数
    final_scores = scores_log_softmax[:, 1].exp().tolist()
    
    return final_scores


# --- [适配后] 主函数 ---

def get_reranked_scores(query, articles, batch_size=4): # 建议根据VRAM大小调整batch_size
    """
    为给定的查询（query）和文章列表（articles）计算重排分数。
    
    Args:
        query (str): 单个查询字符串。
        articles (list[str]): 需要排序的文档字符串列表。
        batch_size (int): 处理时的批量大小。

    Returns:
        list[float]: 与文章列表顺序对应的相关性分数列表。
    """
    start_time = time.time()
    all_scores = []
    
    # 构建 query-document 对
    pairs = [format_instruction(query, doc) for doc in articles]

    # 按批次处理
    for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking"):
        batch_pairs = pairs[i:i+batch_size]
        scores = compute_scores(batch_pairs)
        all_scores.extend(scores)
    
    end_time = time.time()
    return all_scores, end_time - start_time
    
    
if __name__ == "__main__":
    
    query = "diabetes treatment"
    # query = "What is the capital of China?"

    # articles to be ranked for the input query
    articles = [
        "Metformin is a first-line medication for the treatment of type 2 diabetes.",
        "Diabetes mellitus and its chronic complications. Diabetes mellitus is a major cause of morbidity and mortality, and it is a major risk factor for early onset of coronary heart disease. Complications of diabetes are retinopathy, nephropathy, and peripheral neuropathy.",
        "Diagnosis and Management of Central Diabetes Insipidus in Adults. Central diabetes insipidus (CDI) is a clinical syndrome which results from loss or impaired function of vasopressinergic neurons...",
        "Adipsic diabetes insipidus. Adipsic diabetes insipidus (ADI) is a rare but devastating disorder of water balance with significant associated morbidity and mortality...",
        "Nephrogenic diabetes insipidus: a comprehensive overview. Nephrogenic diabetes insipidus (NDI) is characterized by the inability to concentrate urine that results in polyuria and polydipsia...",
        "Impact of Salt Intake on the Pathogenesis and Treatment of Hypertension. Excessive dietary salt (sodium chloride) intake is associated with an increased risk for hypertension..."
    ]
    # articles = [
    #     "The capital of China is Beijing.",
    #     "Gravity"
    # ]

    rerank_scores, duration = get_reranked_scores(query, articles)
    
    # 打印结果，并将文章和分数对应起来
    ranked_results = sorted(zip(articles, rerank_scores), key=lambda x: x[1], reverse=True)
    
    print(f"\nQuery: {query}\n")
    print(f"Reranking took {duration:.4f} seconds.")
    print("--- Reranked Results (Score DESC) ---")
    for doc, score in ranked_results:
        print(f"Score: {score:.4f}\nDocument: {doc[:100]}...\n")