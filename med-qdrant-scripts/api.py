from itertools import chain
import json
from pprint import pprint
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from graph_search import get_graph_docs, SINGLE_GRAPH_TOOLS
from text_search import get_text_docs, SINGLE_TEXT_TOOLS


app = FastAPI()

@app.get("/", response_class=JSONResponse)
@app.get("//", response_class=JSONResponse)
def search_startup(args:str):
    try:
        args = json.loads(args)
        
        all_results = []
        for ar in args:
            tool = ar["source"]
            query = ar["query"]
            retrieval_topk = ar["retrieval_topk"]
            rerank_topk = ar["rerank_topk"]
            assert tool in SINGLE_TEXT_TOOLS + SINGLE_GRAPH_TOOLS, f"{tool} is wrong!"

            if tool in SINGLE_GRAPH_TOOLS:
                # print(query)
                if "," not in query:
                    final_result = []
                else:
                    graph_term, graph_query = query.split(",", maxsplit=1)
                    graph_term, graph_query = graph_term.strip(), graph_query.strip()
                    final_result = get_graph_docs(
                        term=graph_term,
                        query=graph_query,
                        topk=rerank_topk
                    )
            else:
                query = query.strip()
                final_result = get_text_docs(
                    tool=tool,
                    query=query,
                    retrieval_topk=retrieval_topk,
                    rerank_topk=rerank_topk
                )
            all_results.append(final_result)

        return {"success": all_results}

    except Exception as e:
        return repr(e)


if __name__ == "__main__":
    import uvicorn
    assert len(sys.argv) > 1, "PORT?"
    port = int(sys.argv[1])

    uvicorn.run(app, host="0.0.0.0", port=port)
