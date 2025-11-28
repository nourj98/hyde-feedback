import os
import time 
from .index_paths import THE_TOPICS 

def run_bm25(bm25, 
             qids, 
             queries, 
             orig_queries,
             num_hits, 
             corpus_name, 
             query_generator,
             output_filename):
    start_time = time.time()
    bm25_outputs = bm25.run_search(qids, 
                                   queries, 
                                   orig_queries=orig_queries,
                                   k=num_hits, 
                                   return_passage_texts=True,
                                   query_generator=query_generator)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time for search...", elapsed_time, "seconds")
    write_scores_to_file(all_qids=bm25_outputs['qids'],
                         all_docids=bm25_outputs['docids'], 
                         scores=bm25_outputs['bm25_scores'], 
                         output_filename=output_filename)
    print("Quickly evaluating retrieval....", flush=True)
    evaluate(corpus_name, output_filename)
    return bm25_outputs

def write_scores_to_file(all_qids, all_docids, scores, output_filename):
    output_filename = f'{output_filename}'
    with open(output_filename, 'w')  as f:
        rerank_scores = [{'qid': qid, 'docid': docid, 'score': float(score)} for qid, docid, score in zip(all_qids, all_docids, scores)]
        reranked_scores_sorted = sorted(
                rerank_scores,
                key=lambda x: (x['qid'], -x['score'])
            )
        
        rank = 0
        prev_qid = None
        for document in reranked_scores_sorted:
            qid = document["qid"]
            if qid != prev_qid:
                rank = 1
                prev_qid = qid
            else:
                rank += 1
            f.write(f'{qid} Q0 {document["docid"]} {rank} {document["score"]} rank\n')

def evaluate(corpus_name, output_filename):
    # Eval!
    if corpus_name in ['dl21', 'dl22', 'dl23']:
        qrels_name = f'{THE_TOPICS[corpus_name]}-passage'
    else:
        qrels_name = f'{THE_TOPICS[corpus_name]}'

    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {qrels_name} {output_filename}"))
    if 'dl' in corpus_name:
         print(os.system(f"python -m pyserini.eval.trec_eval -c -l 2 -m recall.20 {qrels_name} {output_filename}"))
    else:
        print(os.system(f"python -m pyserini.eval.trec_eval -c -m recall.20 {qrels_name} {output_filename}")) 