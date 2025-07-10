from itertools import chain
import json
from pprint import pprint
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from graph_search import get_graph_docs, SINGLE_GRAPH_TOOLS
from text_search import get_text_docs, get_text_docs_adaptive, SINGLE_TEXT_TOOLS, ADAPTIVE_TEXT_TOOLS
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import traceback

app = FastAPI()

# 创建一个全局的线程池执行器
# 可以根据服务器的核心数和I/O能力调整线程数
executor = ThreadPoolExecutor(max_workers=32)

def process_single_request(ar):
    """
    处理单个检索请求的逻辑，此函数将在线程池中并行执行。
    """
    tool = ar["source"]
    query = ar["query"]
    retrieval_topk = ar["retrieval_topk"]
    rerank_topk = ar["rerank_topk"]
    
    assert tool in SINGLE_TEXT_TOOLS + SINGLE_GRAPH_TOOLS + ADAPTIVE_TEXT_TOOLS, f"{tool} is wrong!"

    final_result = []
    retrieval_time = 0
    rerank_time = 0

    if tool in SINGLE_GRAPH_TOOLS:
        if "," not in query:
            final_result = []
        else:
            graph_term, graph_query = query.split(",", maxsplit=1)
            graph_term, graph_query = graph_term.strip(), graph_query.strip()
            final_result, retrieval_time, rerank_time = get_graph_docs(
                term=graph_term,
                query=graph_query,
                topk=rerank_topk
            )
    elif tool in ADAPTIVE_TEXT_TOOLS:
        query = query.strip()
        final_result, retrieval_time, rerank_time = get_text_docs_adaptive(
            query=query,
            retrieval_topk=retrieval_topk,
            rerank_topk=rerank_topk
        )
    else:
        query = query.strip()
        final_result, retrieval_time, rerank_time = get_text_docs(
            tool=tool,
            query=query,
            retrieval_topk=retrieval_topk,
            rerank_topk=rerank_topk
        )
    
    return final_result, retrieval_time, rerank_time

@app.get("/", response_class=JSONResponse)
@app.get("//", response_class=JSONResponse)
def search_startup(args:str):
    try:
        args_list = json.loads(args)
        
        # 使用线程池并行处理所有请求
        # executor.map会保持原始请求的顺序
        results = list(executor.map(process_single_request, args_list))
        
        all_results = [res[0] for res in results]
        total_retrieval_time = sum(res[1] for res in results)
        total_rerank_time = sum(res[2] for res in results)

        return {
            "success": all_results,
            "timing": {
                "retrieval_time_seconds": total_retrieval_time,
                "rerank_time_seconds": total_rerank_time,
            }
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error processing batch request: {e}\n{error_details}")
        return JSONResponse(status_code=500, content={"detail": repr(e), "traceback": error_details})


if __name__ == "__main__":
    assert len(sys.argv) > 1, "PORT?"
    port = int(sys.argv[1])

    uvicorn.run(app, host="0.0.0.0", port=port)