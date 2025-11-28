import math
from pyserini.search.lucene import querybuilder
from pyserini.analysis import Analyzer, get_lucene_analyzer
import heapq
from pyserini.pyclass import autoclass
from .rocchio_feedback import get_query_vector

JTerm = autoclass('org.apache.lucene.index.Term')
JTermQuery = autoclass('org.apache.lucene.search.TermQuery')

def get_document_vector_hyde(passage, index_reader):
    num_docs = index_reader.stats()['documents']  
    doc_vector = get_query_vector(passage)
    filtered_tf = {}
    for term, freq in doc_vector.items():
        try:
            df, cf = index_reader.get_term_counts(term)
            if 2 <= len(term) <= 20 and df / num_docs <= 0.1:
                filtered_tf[term] = freq
        except:
            next
    
    return filtered_tf
    
def l2_normalize(vec):
    norm = math.sqrt(sum(v**2 for v in vec.values()))
    if norm > 0.001:
        return {t: v / norm for t, v in vec.items()}
    return vec

def prune_top_k(vec, k):
    if k is None or len(vec) <= k:
        return vec
    sorted_items = sorted(vec.items(), key=lambda x: (-x[1], x[0]))
    top_k = sorted_items[:k]
    return dict(top_k)

def compute_mean_vector(vectors, fb_terms=None):
    if not vectors:
        return {}

    vocab = set()
    doc_vecs = []
    norms = []

    for vec in vectors:
        norm = math.sqrt(sum(v**2 for v in vec.values()))
        if norm > 0.001:  
            normed = {t: v / norm for t, v in vec.items()}
            doc_vecs.append(normed)
            vocab.update(normed.keys())
            norms.append(norm)
    
    if not doc_vecs:
        return {}

    mean_vec = {}
    for term in vocab:
        weight = 0.0
        for vec in doc_vecs:
            if term in vec:
                weight += vec[term]
        mean_vec[term] = weight / len(doc_vecs)

    # prune to fb_terms and normalize
    if fb_terms is not None:
        mean_vec = prune_top_k(mean_vec, fb_terms)
    mean_vec = l2_normalize(mean_vec)

    return mean_vec

def hyde_feedback(query_vector, rel_vectors, top_fb_terms=128):
    hyde_vec = compute_mean_vector([query_vector] + rel_vectors, fb_terms=top_fb_terms)
    should = querybuilder.JBooleanClauseOccur["should"].value
    boolean_query_builder = querybuilder.get_boolean_query_builder()
    for term, weight in hyde_vec.items():
        _term = JTermQuery(JTerm("contents", term))
        boost = querybuilder.get_boost_query(_term, weight)
        boolean_query_builder.add(boost, should)
    
    return boolean_query_builder.build()
