# UMLS Database Query Diagrams: https://www.nlm.nih.gov/research/umls/implementation_resources/query_diagrams/er1.html
# sqlite-fts-search-queries: https://saraswatmks.github.io/2020/04/sqlite-fts-search-queries.html

from functools import lru_cache
import json
import os
from pprint import pprint
import re
import sqlite3
import time
import numpy as np
import os, sys
sys.path.append(os.path.abspath('./'))
from utils import get_device_name
from rerank import get_reranked_scores

device = get_device_name()
SINGLE_GRAPH_TOOLS = ["graph"]


class UMLS_Search:
    def __init__(self):
        db_path = '../med-qdrant/umls.sqlite3'
        self.memory_conn = sqlite3.connect(':memory:', check_same_thread=False)
        # load to memory
        file_conn = sqlite3.connect(db_path)
        file_conn.backup(self.memory_conn)
        file_conn.close()
        # set to only-read mode
        self.memory_conn.execute('PRAGMA query_only = ON')
        self.memory_conn.execute('PRAGMA synchronous = OFF')
        self.memory_conn.execute('PRAGMA journal_mode = OFF')
        self.memory_conn.execute('PRAGMA temp_store = MEMORY')
        
        self.cui_to_names = {}
        res = self.memory_conn.cursor().execute(f'SELECT * FROM MRCONSOEM').fetchall()
        for i in res:
            cui = i[0]
            name = i[2]
            if cui not in self.cui_to_names:
                self.cui_to_names[cui] = [set(), set()]
            if name.lower() not in self.cui_to_names[cui][1]:
                self.cui_to_names[cui][0].add(name)
                self.cui_to_names[cui][1].add(name.lower())
        self.cui_to_names = {k: sorted(list(v[0])) for k, v in self.cui_to_names.items()}
        

    def term_to_cui(self, term):
        # EXACT MATCH "COLLATE NOCASE" is set when creating the table!
        term = term.replace("\"", " ").strip()
        res = self.memory_conn.cursor().execute(f'SELECT * FROM MRCONSOEM WHERE STR="{term}" LIMIT 1').fetchone()
        if res is None:
            # FUZZY MATCH
            term = term.replace("'", " ").strip()
            res = self.memory_conn.cursor().execute(f'SELECT * FROM MRCONSO WHERE STR MATCH \'"{term}"\' ORDER BY rank LIMIT 1').fetchone()
        if res is not None:
            cui = res[0]
            return cui
        return None
    
    def cui_to_definition(self, cui):
        res = self.memory_conn.cursor().execute(f'SELECT * FROM MRDEF WHERE CUI="{cui}"').fetchall()
        if res is not None:
            msh_def = None
            nci_def = None
            icf_def = None
            csp_def = None
            hpo_def = None
            other_def = None
            for definition in res:
                source = definition[1]
                if source == "MSH":
                    msh_def = definition[2]
                    break
                elif source == "NCI":
                    nci_def = definition[2]
                elif source == "ICF":
                    icf_def = definition[2]
                elif source == "CSP":
                    csp_def = definition[2]
                elif source == "HPO":
                    hpo_def = definition[2]
                else:
                    other_def = definition[2]
            defi = msh_def or nci_def or icf_def or csp_def or hpo_def or other_def
            return defi
        return None
    
    def cui_to_relations(self, cui):
        res = self.memory_conn.cursor().execute(f'SELECT STR2,RELA,STR1 FROM MRREL WHERE CUI1="{cui}" OR CUI2="{cui}"').fetchall()
        if res is not None:
            res = list(set(res))        
        return res

class DrugBank_Search:
    def __init__(self):
        self.data = json.load(open("../med-qdrant/drugbank_info.json"))
    
    def name_to_info(self, name):
        name = name.lower()
        if name in self.data:
            return self.data[name]
        else:
            return ""


umls_search = UMLS_Search()
drugbank_search = DrugBank_Search()

@lru_cache(maxsize=100000000)
def get_graph_docs(term, query, topk):
    start_time = time.time()
    cui = umls_search.term_to_cui(term)
    if cui is not None:
        # 1. search
        definition = umls_search.cui_to_definition(cui) or "" + drugbank_search.name_to_info(term) # type: ignore
        rels=umls_search.cui_to_relations(cui)
        retrieval_time = time.time() - start_time

        # 2. rerank
        rel_texts = [f"{rel[0]} {rel[1]} {rel[2]}" for rel in rels]
        scores, rerank_time = get_reranked_scores(
            query=query,
            articles=rel_texts
        )
        zipped_score_rel = list(zip(scores, rels))
        zipped_score_rel.sort(key=lambda x: x[0], reverse=True)
        rerank_rels = [i[1] for i in zipped_score_rel[:topk]]

        relation = "; ".join([f"({rel[0]}, {rel[1]}, {rel[2]})" for rel in rerank_rels])
        para_text = f"Definition: {definition}\n" if definition else ""
        para_text += f"Relation: {relation}." if relation else ""

        if para_text:
            results = [{"title": "/".join(umls_search.cui_to_names[cui]), "para": para_text, "dataset": "umls"}]
            # Manually add rerank scores to graph results for consistency
            if results and zipped_score_rel:
                results[0]['rerank_score'] = zipped_score_rel[0][0]
            return results, retrieval_time, rerank_time
            
    return [], 0, 0


if __name__ == "__main__":
    results, retrieval_time, rerank_time = get_graph_docs(term="Oxamniquine", query="what is the inhibitor of ASPIRIN?", topk=10)
    print(results)
    print(f"Retrieval Time: {retrieval_time:.4f}s")
    print(f"Rerank Time: {rerank_time:.4f}s")