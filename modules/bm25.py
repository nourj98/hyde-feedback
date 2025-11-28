import json
from tqdm import tqdm 

class BM25():
    def __init__(self, searcher, task):
        self.searcher = searcher
        self.task = task

    def load_document_text(self, doc_id):
        # simplify this a bit...
        content = json.loads(self.searcher.doc(doc_id).raw())
        if self.task in ['dl19', 'dl20']:
            text = content['contents']
        elif self.task == 'dl21' or self.task == 'dl22' or self.task == 'dl23':
            text = content['passage']
        elif 'bright' in self.task:
            text = content['contents']
        else:
            text = content['text']
            if 'title' in content:
                text = f'{content["title"]} {text}'
        assert (text != '')
        return text 

    def run_search(self, qids, queries, orig_queries, k=1000, return_passage_texts=False, query_generator=None):
        # TODO: can prob do this in a batch search...
        all_docids = []
        all_passage_texts = []
        all_queries = [] 
        all_qids = [] 
        all_bm25_scores = []
        for idx in tqdm(range(len(queries))):
            qid = qids[idx]
            query = queries[idx]
            orig_query = orig_queries[idx]
            hits = self.searcher.search(query, k=k, query_generator=query_generator)
            hits = [hit for hit in hits if hit.docid != qid]
            all_docids += [hit.docid for hit in hits]
            all_bm25_scores += [hit.score for hit in hits]
            all_queries += [orig_query]*len(hits)
            all_qids += [qid]*len(hits)

            if return_passage_texts:
                passages = [self.load_document_text(hit.docid) for hit in hits]
                all_passage_texts += passages
    
        return {'qids': all_qids,
                'queries': all_queries,
                'docids': all_docids,
                'passage_texts': all_passage_texts,
                'bm25_scores': all_bm25_scores,
                }