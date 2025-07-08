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
            assert source in SEARCH_ACTION_DESC
            for q in queries:
                args.append({"source": source, "query": q, "retrieval_topk": 2*self.topk, "rerank_topk": self.topk})
        ##### Run Search #####
        try_number = 10
        for try_index in range(try_number):
            try:
                params = {
                    "args": json.dumps(args, ensure_ascii=False)
                }
                encoded_params = urllib.parse.urlencode(params)
                search_url = f"http://127.0.0.1:10002/?{encoded_params}"
                print(search_url)
                t1 = time.time()
                search_result = session.get(search_url, timeout=300).content.decode('utf-8')
                print('single:', time.time() - t1)
                search_result = json.loads(search_result)["success"]
                assert len(search_result) == len(args)
                break
                
            except Exception as e:
                if try_index == try_number - 1:
                    raise ValueError(f"Error in Search: {search_url} Error: {e}")
                print(f"Error in Search: {search_url} Error: {e}")
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
    units = [
    [
        "book",
        [
        "fenofibrate and sleep apnoea syndrome",
        "proof of concept studies in sleep disorders",
        "role of fenofibrate in respiratory disorders"
        ]
    ],
    [
        "guideline",
        [
        "fenofibrate in sleep apnoea syndrome",
        "clinical guidelines for sleep apnoea treatment",
        "use of fenofibrate in respiratory conditions"
        ]
    ],
    [
        "research",
        [
        "fenofibrate and sleep apnoea syndrome",
        "proof of concept studies on fenofibrate for sleep disorders",
        "efficacy of fenofibrate in treating sleep apnoea"
        ]
    ],
    [
        "wiki",
        [
        "fenofibrate and sleep apnoea",
        "proof of concept studies in sleep disorders",
        "role of fenofibrate in sleep"
        ]
    ],
    [
        "graph",
        [
        "fenofibrate , role in sleep apnoea",
        "sleep apnoea syndrome , treatment options",
        "proof of concept , medical studies"
        ]
    ]
    ]
    # [
    #     ["research", ["HIV", "123"]],
    #     ["book", ["HIV", "123"]],
    #     ["graph", ["HIV , medicine", "123"]],
    # ]
    print(json.dumps(retriever.run(units), indent=2))