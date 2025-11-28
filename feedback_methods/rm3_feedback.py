import math
from pyserini.search.lucene import querybuilder
from pyserini.analysis import Analyzer, get_lucene_analyzer
import heapq
from pyserini.pyclass import autoclass

JTerm = autoclass('org.apache.lucene.index.Term')
JTermQuery = autoclass('org.apache.lucene.search.TermQuery')

def prune_top_k(vec, k):
    if k is None or len(vec) <= k:
        return vec
    sorted_items = sorted(vec.items(), key=lambda x: (-x[1], x[0]))
    top_k = sorted_items[:k]
    return dict(top_k)

def l1_normalize(vec):
    total = sum(abs(v) for v in vec.values())
    return {t: v / total for t, v in vec.items()} if total > 0 else vec

def get_document_vector_rm3(docid, index_reader, filter_terms=False):
    num_docs = index_reader.stats()['documents']
    doc_vector = index_reader.get_document_vector(docid)
    filtered_tf = {}

    for term, freq in doc_vector.items():
        if filter_terms and not term.isalnum():
            continue  # skip if not [a-z0-9]+
        try:
            df, cf = index_reader.get_term_counts(term)
            if 2 <= len(term) <= 20 and df / num_docs <= 0.1:
                filtered_tf[term] = freq
        except Exception:
            continue

    return filtered_tf

def compute_relevance_model(vectors, document_scores, fb_terms=None):
    if not vectors:
        return {}

    vocab = set()
    doc_vecs = []
    doc_scores = []
    norms = []
    for idx, vec in enumerate(vectors):
        if fb_terms is not None:
            vec = prune_top_k(vec, fb_terms)        
        
        norm = sum(abs(v) for v in vec.values())
        if norm > 0.001:
            doc_vecs.append(vec)
            vocab.update(vec.keys())
            norms.append(norm)
            doc_scores.append(document_scores[idx])

    if not doc_vecs:
        return {}

    rm3_vec = {}
    for term in vocab:
        fb_weight = 0.0
        for i, vec in enumerate(doc_vecs):
            if term in vec:
                fb_weight += (vec[term] / norms[i]) * doc_scores[i]
        rm3_vec[term] = fb_weight

    if fb_terms is not None:
        rm3_vec = prune_top_k(rm3_vec, fb_terms)
    rm3_vec = l1_normalize(rm3_vec)

    return rm3_vec


def interpolate(query_vector, document_vectors, query_weight):
    vocab = set(query_vector.keys()) | set(document_vectors.keys())
    rm3_vec = {}
    for term in vocab:
        score = (
            query_weight * query_vector.get(term, 0.0)
            + (1 - query_weight) * document_vectors.get(term, 0.0)
        )
        if score > 0:
            rm3_vec[term] = score
    return rm3_vec

def rm3_feedback(
    query_vector,
    rel_vectors,
    document_scores,
    original_query_weight=0.5,
    top_fb_docs=10,
    top_fb_terms=128,
):
    query_vector = l1_normalize(query_vector)
    if top_fb_docs is not None:
        rel_vectors = rel_vectors[:top_fb_docs]

    rel_model = compute_relevance_model(
        rel_vectors, document_scores, fb_terms=top_fb_terms
    )

    rm3_vec = interpolate(query_vector, rel_model, original_query_weight)

    should = querybuilder.JBooleanClauseOccur["should"].value
    boolean_query_builder = querybuilder.get_boolean_query_builder()
    for term, weight in rm3_vec.items():
        _term = JTermQuery(JTerm("contents", term))
        boost = querybuilder.get_boost_query(_term, weight)
        boolean_query_builder.add(boost, should)
    return boolean_query_builder.build()