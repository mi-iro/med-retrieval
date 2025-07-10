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
    "adaptive_text":"The API provides access to all text-based sources (book, guideline, research, wiki) with a unified ranking.",
}
SEARCH_ACTION_PARAM = {
    "book":         r"{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to 3 queries)",
    "guideline":    r"{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to 3 queries)",
    "research":     r"{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to 3 queries)",
    "wiki":         r"{search_query0} ; {search_query1} ; ... (Use ; to separate the queries, 0 to 3 queries)",
    "graph":        r"{medical_term0} , {query_for_term0} ; {medical_term1} , {query_for_term1} ; ... (Use ; to separate the queries, 0 to 3 queries. Each query should use , to separate the {medical_term} and {query_for_term})",
    "adaptive_text":r"{search_query}"
}

session = requests.Session()

class Retriever:
    def __init__(self, topk):
        self.topk = topk

    def run(self, source_and_queries, add_query=False, adaptive=False):
        args = []
        if adaptive:
            # 自适应模式：使用第一个查询作为统一查询
            # source_and_queries 格式为: [["adaptive_text", ["your query"]]]
            main_query = source_and_queries[0][1][0]
            args.append({"source": "adaptive_text", "query": main_query, "retrieval_topk": 2 * self.topk, "rerank_topk": self.topk})
        else:
            # 常规模式
            for source, queries in source_and_queries:
                if not queries:
                    continue
                assert source in SEARCH_ACTION_DESC
                for q in queries:
                    args.append({"source": source, "query": q, "retrieval_topk": 2*self.topk, "rerank_topk": self.topk})
        
        if not args:
            return [], {}

        ##### Run Search #####
        timing_info = {}
        try_number = 10
        for try_index in range(try_number):
            try:
                params = {
                    "args": json.dumps(args, ensure_ascii=False)
                }
                encoded_params = urllib.parse.urlencode(params)
                search_url = f"http://127.0.0.1:10002/?{encoded_params}"
                print(f"Executing search URL: {search_url}")
                
                response = session.get(search_url, timeout=300)
                response.raise_for_status()
                
                response_json = response.json()
                search_result = response_json.get("success", [])
                timing_info = response_json.get("timing", {})

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

            docs_for_query = search_result[index]
            if len(docs_for_query) > 0:
                # Attach rerank_score to the original arg dict
                # This makes it easier to access later
                ar['results'] = docs_for_query
                single_text += "\n".join([f"(Title: {doc.get('title', 'N/A')}) {doc.get('para', '')}" 
                                for doc in docs_for_query])
            else:
                ar['results'] = []
                single_text += "There are no searching results."

            single_text = single_text.strip()
            ar["docs"] = single_text

        return args, timing_info


if __name__ == "__main__":
    retriever = Retriever(topk=10)
    
    # --- 原有测试 ---
    units_standard = [
        ["book", ["fenofibrate and sleep apnoea syndrome"]],
        ["guideline", ["fenofibrate in sleep apnoea syndrome"]],
        ["research", ["efficacy of fenofibrate in treating sleep apnoea"]],
        ["wiki", ["fenofibrate and sleep apnoea"]],
        ["graph", ["fenofibrate , role in sleep apnoea"]],
    ]
    print("\n--- Running Standalone Example (Standard Mode) ---")
    retrieved_results_std, timings_std = retriever.run(units_standard)
    print("\n--- Retrieved Documents (Standard) ---")
    print(json.dumps(retrieved_results_std, indent=2))
    print("\n--- Timing Information (Standard) ---")
    print(json.dumps(timings_std, indent=2))
    
    # --- 新增自适应模式测试 ---
    units_adaptive = [
        ["adaptive_text", ["fenofibrate and sleep apnoea syndrome"]]
    ]
    print("\n--- Running Standalone Example (Adaptive Mode) ---")
    retrieved_results_adapt, timings_adapt = retriever.run(units_adaptive, adaptive=True)
    print("\n--- Retrieved Documents (Adaptive) ---")
    print(json.dumps(retrieved_results_adapt, indent=2))
    print("\n--- Timing Information (Adaptive) ---")
    print(json.dumps(timings_adapt, indent=2))