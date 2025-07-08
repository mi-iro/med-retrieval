import json
import re
import os
import sys
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.append(os.path.abspath("./"))

# VLLMChatLLM 类定义保持不变
class VLLMChatLLM():
    def __init__(self, llm_name):
        assert "CUDA_VISIBLE_DEVICES" in os.environ
        self.llm = LLM(
            model=llm_name,
            enable_prefix_caching=True,
            max_model_len=32000,
            max_num_seqs=1
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

    def run(self, prompt):
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        sample_params = SamplingParams(
            n=1,
            temperature=0,
            stop=[],
            include_stop_str_in_output=True,
            max_tokens=1000,
            seed=0
        )
        response = self.llm.generate(
            prompts=[prompt],
            sampling_params=sample_params,
            use_tqdm=False,
        )
        output = response[0].outputs[0].text
        print(prompt)
        print('-'*89)
        print(output)
        print('='*89)
        return output

# 常量定义保持不变
GENERATE_QUERIES_TEMPLATE = """To answer the question labeled as # Question, please construct appropriate queries to get the information you need.
1. Please give the search queries following the format in # Query Format. The source can have up to {N_QUERIES} queries, separated by `;`. Please ensure the diversity of queries from the same source. For each source, if you think no information retrieval is needed, simply output an empty tag for that source, for example: <book></book>.
2. The queries for the source should accurately reflect the specific information needs from that source.
# Question
{question}
# Source Description
book: The API provides access to medical knowledge resource including various educational resources and textbooks.
guideline: The API provides access to clinical guidelines from leading health organizations.
research: The API provides access to advanced biomedical research, facilitating access to specialized knowledge and resources.
wiki: The API provides access to general knowledge across a wide range of topics.
graph: The API provides a structured knowledge graph that connects medical definitions and related terms.
# Query Format
<book>{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to {N_QUERIES} queries)</book>
<guideline>{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to {N_QUERIES} queries)</guideline>
<research>{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to {N_QUERIES} queries)</research>
<wiki>{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to {N_QUERIES} queries)</wiki>
<graph>{medical_term0} , {query_for_term0} ; {medical_term1} , {query_for_term1} ; ... (Use ; to separate the queries, 0 to {N_QUERIES} queries. Each query should use , to separate the {medical_term} and {query_for_term})</graph>"""

# --- 服务启动时执行一次模型初始化 ---
print("Initializing VLLM Model for query generation...")
llm = VLLMChatLLM(llm_name="model/Qwen2.5-72B-Instruct-AWQ")
print("VLLM Model initialized successfully.")

# --- 封装成可被外部调用的函数（已更新） ---
def generate_queries_for_question(question: str, n_queries: int) -> tuple[list, float]:
    """
    根据输入的问题，调用大语言模型生成结构化的子查询。

    Args:
        question: 用户提出的原始问题字符串。
        n_queries: 每个数据源最多生成的查询数量。

    Returns:
        一个元组，包含针对不同数据源的查询指令列表和生成查询所花费的时间。
    """
    start_time = time.time()
    # 使用传入的 n_queries 参数替换模板中的 {N_QUERIES}
    generate_queries_prompt = GENERATE_QUERIES_TEMPLATE.replace("{question}", question).replace("{N_QUERIES}", str(n_queries))
    generate_queries_output = llm.run(generate_queries_prompt)

    source_and_queries = []
    for source in ["book", "guideline", "research", "wiki", "graph"]:
        pattern = f'<{source}>(.*?)</{source}>'
        match = re.search(pattern, generate_queries_output, re.DOTALL)
        queries = ""
        if match is not None:
            queries = match.group(1).strip()
            # 清理占位符和说明性文本
            replacements = {
                "{search_query0} ; {search_query1} ; ...": "",
                f"(Use ; to separate the queries, 0 to {n_queries} queries)": "",
                "{": "",
                "}": ""
            }
            for old, new in replacements.items():
                queries = queries.replace(old, new)

        # 使用传入的 n_queries 参数来分割字符串
        queries_list = [q.strip() for q in queries.split(";", maxsplit=n_queries - 1) if q.strip()]
        
        final_queries = []
        for q in queries_list:
            if source == "graph":
                final_queries.append(q.replace(";", ','))
            else:
                final_queries.append(q)

        source_and_queries.append([source, final_queries])
    
    end_time = time.time()
    return source_and_queries, end_time - start_time


# --- 保留原有的 __main__ 块，以便该脚本可以独立运行进行测试 ---
if __name__ == "__main__":
    example_question = """Proof of concept study: does fenofibrate have a role in sleep apnoea syndrome?
A. yes
B. no
C. maybe"""
    
    print("\n--- Running Standalone Example ---")
    # 使用默认值进行测试
    generated_data, planning_time = generate_queries_for_question(example_question, n_queries=3)
    print(f"\nQuery planning took {planning_time:.4f} seconds.")
    print("\n--- Generated Queries ---")
    print(json.dumps(generated_data, indent=2))