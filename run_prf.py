import os 
import argparse
from tqdm import tqdm 

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

from feedback_methods.rocchio_feedback import *
from feedback_methods.hyde_feedback import *
from feedback_methods.rm3_feedback import *

from modules.index_paths import THE_SPARSE_INDEX, load_queries_qids
from modules.helpers import evaluate 

K1=0.9
B=0.4

def generate_avg_vector_query(query, hits, top_fb_docs):
    relevant_docids = [hit.docid for hit in hits]
    scores = [hit.score for hit in hits]
    rel_feedback_vectors = [get_document_vector_rocchio(docid, index_reader) for docid in relevant_docids]
    hyde_query = hyde_feedback(query_vector=get_query_vector(query), 
                               rel_vectors=rel_feedback_vectors)
        
    return hyde_query

def generate_rm3_query(query, hits, top_fb_docs):
    relevant_docids = [hit.docid for hit in hits]
    scores = [hit.score for hit in hits]
    rel_feedback_vectors = [get_document_vector_rm3(docid, index_reader, True) for docid in relevant_docids]
    rm3_query = rm3_feedback(query_vector=get_query_vector(query), 
                    rel_vectors=rel_feedback_vectors,
                    document_scores=scores,
                    top_fb_docs=top_fb_docs)
    return rm3_query

def generate_rocchio_query(query, hits, top_fb_docs):
    relevant_docids = [hit.docid for hit in hits]
    rel_feedback_vectors = [get_document_vector_rocchio(docid, index_reader) for docid in relevant_docids]
    rocchio_query = rocchio_feedback(get_query_vector(query),
                                     rel_vectors=rel_feedback_vectors,
                                     nrel_vectors=None,
                                     top_fb_docs=top_fb_docs) 
    return rocchio_query

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Custom PRF w/ BM25')
    parser.add_argument('--corpus_name', required=True, help='dataset path')
    parser.add_argument('--feedback_documents', type=int, default=8, help='search results that PRF uses')
    parser.add_argument('--returned_hits', type=int, default=100, help='search results for feedback reranking')
    parser.add_argument('--feedback_mechanism', default=None, type=str)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    ##################################################################
    # Define and load searcher and index readers
    qids, queries = load_queries_qids(args.corpus_name)
    index_reader = LuceneIndexReader.from_prebuilt_index(THE_SPARSE_INDEX[args.corpus_name])
    lucene_bm25_searcher = LuceneSearcher.from_prebuilt_index(THE_SPARSE_INDEX[args.corpus_name])
    lucene_bm25_searcher.set_bm25(k1=K1, b=B)
    if args.feedback_mechanism is None:
        output_filename = os.path.join(args.output_folder, f'bm25.trec')
    else:
        output_filename = os.path.join(args.output_folder, f'{args.feedback_mechanism}_python_impl.trec')
    ##################################################################
    # Run retrieval
    os.makedirs(args.output_folder, exist_ok=True)
    with open(output_filename, 'w')  as f:
       for idx in tqdm(range(len(queries))):
            qid = qids[idx]
            query = queries[idx]
            if args.feedback_mechanism is None:
                final_hits = lucene_bm25_searcher.search(query, k=args.returned_hits) 
                for rank, hit in enumerate(final_hits):
                    f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} bm25\n')
            else:
                hits = lucene_bm25_searcher.search(query, k=args.feedback_documents) 
                if args.feedback_mechanism == 'rm3':
                    feedback_query = generate_rm3_query(query, hits, top_fb_docs=args.feedback_documents)
                elif args.feedback_mechanism == 'hyde':
                    feedback_query = generate_avg_vector_query(query, hits, top_fb_docs=args.feedback_documents)
                else:
                    feedback_query = generate_rocchio_query(query, hits, top_fb_docs=args.feedback_documents) 
                final_hits = lucene_bm25_searcher.search(feedback_query, k=args.returned_hits)
                for rank, hit in enumerate(final_hits):
                    f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} feedback_python_impl\n')
    
    evaluate(args.corpus_name, output_filename)  