# 此文件内容与您提供的 run_retriever.py 完全相同。
# 只需将文件名从 run_retriever.py 改为 retriever.py 即可。
import json
import os
import sys
import time
import numpy as np
import requests
import urllib
sys.path.append(os.path.abspath("./"))


SEARCH_ACTION_DESC = {
    "book":         "The API provides access to medical knowledge resource including various educational resources and textbooks.",
    "guideline":    "The API provides access to clinical guidelines from leading health organizations.",
    "research":     "The API provides access to advanced biomedical research, facilitating access to specialized knowledge and resources.",
    "wiki":         "The API provides access to general knowledge across a wide range of topics.",
    "graph":        "The API provides a structured knowledge graph that connects medical definitions and related terms.",
}
SEARCH_ACTION_PARAM = {
    "book":         r"{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to 3 queries)",
    "guideline":    r"{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to 3 queries)",
    "research":     r"{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to 3 queries)",
    "wiki":         r"{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to 3 queries)",
    "graph":        r"{medical_term0} , {query_for_term0} ; {medical_term1} , {query_for_term1} ; ... (Use ; to separate the queries, 0 to 3 queries. Each query should use , to separate the {medical_term} and {query_for_term})",
}

session = requests.Session()

class Retriever:
    def __init__(self, topk):
        self.topk = topk

    def run(self, source_and_queries, add_query=False):
        args = []
        for source, queries in source_and_queries:
            if not queries:  # 如果查询列表为空，则跳过
                continue
            assert source in SEARCH_ACTION_DESC
            for q in queries:
                args.append({"source": source, "query": q, "retrieval_topk": 2*self.topk, "rerank_topk": self.topk})
        
        if not args: # 如果没有有效的查询，直接返回空结果
            return []

        ##### Run Search #####
        try_number = 10
        for try_index in range(try_number):
            try:
                params = {
                    "args": json.dumps(args, ensure_ascii=False)
                }
                encoded_params = urllib.parse.urlencode(params)
                search_url = f"http://127.0.0.1:10002/?{encoded_params}"
                print(f"Executing search URL: {search_url}")
                t1 = time.time()
                response = session.get(search_url, timeout=300)
                response.raise_for_status() # 检查请求是否成功
                search_result = response.json()["success"]
                print(f'Search execution time: {time.time() - t1:.2f}s')
                assert len(search_result) == len(args)
                break
                
            except Exception as e:
                print(f"Attempt {try_index + 1}/{try_number} failed. Error in Search: {search_url} Error: {e}")
                if try_index == try_number - 1:
                    raise ValueError(f"Error in Search: {search_url} Error: {e}")
                time.sleep(6)
        ######################
        for index, ar in enumerate(args):
            if add_query:
                single_text = f"## source: {ar['source']}; query: {ar['query']}\n"
            else:
                single_text = f"## source: {ar['source']}\n"
            if len(search_result[index]) > 0:
                single_text += "\n".join([f"(Title: {doc['title']}) {doc['para']}" 
                                for doc in search_result[index]])
            else:
                single_text += "There are no searching results."
            single_text = single_text.strip()
            ar["docs"] = single_text

        return args


if __name__ == "__main__":
    retriever = Retriever(topk=10)
    # 示例用法保持不变
    units = [
        ["book", ["fenofibrate and sleep apnoea syndrome"]],
        ["guideline", ["fenofibrate in sleep apnoea syndrome"]],
        ["research", ["efficacy of fenofibrate in treating sleep apnoea"]],
        ["wiki", ["fenofibrate and sleep apnoea"]],
        ["graph", ["fenofibrate , role in sleep apnoea"]],
    ]
    print("\n--- Running Standalone Example ---")
    retrieved_results = retriever.run(units)
    print("\n--- Retrieved Documents ---")
    print(json.dumps(retrieved_results, indent=2))