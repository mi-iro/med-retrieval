from itertools import chain
import json
from pprint import pprint
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from graph_search import get_graph_docs, SINGLE_GRAPH_TOOLS
from text_search import get_text_docs, get_text_docs_adaptive, SINGLE_TEXT_TOOLS, ADAPTIVE_TEXT_TOOLS


app = FastAPI()

@app.get("/", response_class=JSONResponse)
@app.get("//", response_class=JSONResponse)
def search_startup(args:str):
    try:
        args = json.loads(args)
        
        all_results = []
        total_retrieval_time = 0
        total_rerank_time = 0

        for ar in args:
            tool = ar["source"]
            query = ar["query"]
            retrieval_topk = ar["retrieval_topk"]
            rerank_topk = ar["rerank_topk"]
            # 更新断言，加入ADAPTIVE_TEXT_TOOLS
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
            # 新增分支：处理自适应检索模式
            elif tool in ADAPTIVE_TEXT_TOOLS:
                query = query.strip()
                final_result, retrieval_time, rerank_time = get_text_docs_adaptive(
                    query=query,
                    retrieval_topk=retrieval_topk,
                    rerank_topk=rerank_topk
                )
            # 原有文本检索逻辑
            else:
                query = query.strip()
                final_result, retrieval_time, rerank_time = get_text_docs(
                    tool=tool,
                    query=query,
                    retrieval_topk=retrieval_topk,
                    rerank_topk=rerank_topk
                )

            all_results.append(final_result)
            total_retrieval_time += retrieval_time
            total_rerank_time += rerank_time

        return {
            "success": all_results,
            "timing": {
                "retrieval_time_seconds": total_retrieval_time,
                "rerank_time_seconds": total_rerank_time,
            }
        }

    except Exception as e:
        return repr(e)


if __name__ == "__main__":
    import uvicorn
    assert len(sys.argv) > 1, "PORT?"
    port = int(sys.argv[1])

    uvicorn.run(app, host="0.0.0.0", port=port)