import os 
import argparse

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

from feedback_methods.rocchio_feedback import *
from feedback_methods.hyde_feedback import *
from feedback_methods.rm3_feedback import *

from modules.bm25 import BM25
from modules.hyde import Query2Doc
from modules.helpers import run_bm25
from modules.index_paths import THE_SPARSE_INDEX, load_queries_qids

K1=0.9
B=0.4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate HyDE')
    parser.add_argument('--model_path', required=True, help='base model path')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--corpus_name', required=True, help='dataset path')
    parser.add_argument('--returned_hits', type=int, default=100, help='search hits after feedback reranking')
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
    output_filename = f'q2d_model-{os.path.basename(args.model_path)}.trec'    
    q2d = Query2Doc(base_model_name_or_path=args.model_path, 
                    num_gpus=args.num_gpus, 
                    dataset_prompt=args.corpus_name) 
    synthetic_passages = q2d.predict(queries=queries, task=args.corpus_name)
    expanded_queries = [f'{queries[idx]} {queries[idx]} {queries[idx]} {queries[idx]} {queries[idx]} {passage}' \
                            for idx, passage in enumerate(synthetic_passages)]   

    #######################################
    # Run initial retrieval
    os.makedirs(args.output_folder, exist_ok=True)
    bm25_output_filename = os.path.join(args.output_folder, output_filename)    
    bm25_outputs = run_bm25(bm25=bm25, 
                            qids=qids, 
                            queries=expanded_queries, 
                            orig_queries=queries,
                            num_hits=args.returned_hits,
                            corpus_name=args.corpus_name, 
                            query_generator=None,
                            output_filename=bm25_output_filename)