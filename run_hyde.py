import os 
import math 
import argparse

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

from feedback_methods.rocchio_feedback import *
from feedback_methods.hyde_feedback import *
from feedback_methods.rm3_feedback import *

from modules.bm25 import BM25
from modules.hyde import HyDE
from modules.helpers import run_bm25
from modules.index_paths import THE_SPARSE_INDEX, load_queries_qids


K1=0.9
B=0.4

def get_bm25_vector(passage):
    index_stats = index_reader.stats()
    num_docs = index_stats['documents']
    avg_length = index_stats['total_terms'] / num_docs

    doc_vector = get_query_vector(passage) # {term: freq}
    doc_length = sum(doc_vector.values())
    bm25_vector = {}
    for term, freq in doc_vector.items():
        # We only consider terms in our corpus...
        try:
            df, cf = index_reader.get_term_counts(term)
            idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
            score = idf * (freq / (freq + K1 * (1 - B + B * ( doc_length  /  avg_length))))
            bm25_vector[term] = score
        except:
            continue
    
    return bm25_vector

def compute_bm25_score(query, passage):
    # follows: https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md
    query_vector = get_query_vector(query)
    multihot_query_weights = {term: 1 for term in query_vector.keys()}
    bm25_weights = get_bm25_vector(passage)
    score = sum({term: bm25_weights[term] \
                    for term in bm25_weights.keys() & \
                    multihot_query_weights.keys()}.values())
    return score

def generate_hyde_query_from_feedback(query, synthetic_passages, feedback_mechanism, index_reader, corpus_name):
    synthetic_passage_vectors = [get_document_vector_hyde(p, index_reader) for p in synthetic_passages]
    query_vector = get_query_vector(query)
    if feedback_mechanism == 'rocchio':
        q_new = rocchio_feedback(query_vector=query_vector, 
                                 rel_vectors=synthetic_passage_vectors,
                                 nrel_vectors=None,
                                 gamma=0)
    elif feedback_mechanism == 'rm3':
        # For simplicity and to follow Anserini, setting document scores as the BM25 score
        document_scores = [compute_bm25_score(query, p) for p in synthetic_passages]
        q_new = rm3_feedback(query_vector=query_vector,
                             rel_vectors=synthetic_passage_vectors,
                             document_scores=document_scores)    
    elif feedback_mechanism == 'hyde':
        q_new = hyde_feedback(query_vector=query_vector,
                              rel_vectors=synthetic_passage_vectors)
    elif feedback_mechanism == 'mugi':
        # This directly implements adaptive ratio from 
        # https://github.com/lezhang7/Retrieval_MuGI/tree/main
        repetition_ratio=5 
        gen_ref =  " ".join(synthetic_passages)
        repetition_times = (len(gen_ref)//len(query))//repetition_ratio
        q_new = (query + ' ')*repetition_times + gen_ref
    else:
       q_new = " ".join(synthetic_passages) + " " + query
    return q_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate HyDE')
    parser.add_argument('--model_path', required=True, help='base model path')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--corpus_name', required=True, help='dataset path')
    parser.add_argument('--returned_hits', type=int, default=100, help='search hits after feedback reranking')
    parser.add_argument('--feedback_mechanism', type=str)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    ##################################################################
    # Define and load searcher and index readers
    qids, queries = load_queries_qids(args.corpus_name)
    index_reader = LuceneIndexReader.from_prebuilt_index(THE_SPARSE_INDEX[args.corpus_name])
    lucene_bm25_searcher = LuceneSearcher.from_prebuilt_index(THE_SPARSE_INDEX[args.corpus_name])
    lucene_bm25_searcher.set_bm25(k1=K1, b=B)
    bm25 = BM25(searcher=lucene_bm25_searcher, task=args.corpus_name)
    ##################################################################
    # Create HyDE expansions and generate new improved queries
    output_filename = f'hyde_model-{os.path.basename(args.model_path)}_feedback-{args.feedback_mechanism}.trec'    
    hyde = HyDE(base_model_name_or_path=args.model_path, 
                num_gpus=args.num_gpus, 
                dataset_prompt=args.corpus_name) 
    synthetic_passages = hyde.predict(queries=queries, task=args.corpus_name)
    expanded_queries = [generate_hyde_query_from_feedback(queries[idx], 
                                                          passages, 
                                                          args.feedback_mechanism, 
                                                          index_reader, 
                                                          args.corpus_name)
                            for idx, passages in enumerate(synthetic_passages)]     
    ##################################################################
    # Run retrieval
    os.makedirs(args.output_folder, exist_ok=True)
    bm25_output_filename = os.path.join(args.output_folder, output_filename)    
    query_generator = None
    bm25_outputs = run_bm25(bm25=bm25, 
                            qids=qids, 
                            queries=expanded_queries, 
                            orig_queries=queries,
                            num_hits=args.returned_hits,
                            corpus_name=args.corpus_name, 
                            query_generator=query_generator,
                            output_filename=bm25_output_filename)