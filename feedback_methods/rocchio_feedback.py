import math
from pyserini.search.lucene import querybuilder
from pyserini.analysis import Analyzer, get_lucene_analyzer
import heapq
from pyserini.pyclass import autoclass

JTerm = autoclass('org.apache.lucene.index.Term')
JTermQuery = autoclass('org.apache.lucene.search.TermQuery')

def get_query_vector(query):
    analyzer = Analyzer(get_lucene_analyzer(stemmer='porter', stopwords=True))
    query_tokens = analyzer.analyze(query)

    query_term_weights = {}
    for k in query_tokens:
        query_term_weights[k] = query_term_weights.get(k, 0) + 1
    return query_term_weights

def get_document_vector_rocchio(docid, index_reader):
    num_docs = index_reader.stats()['documents']  
    doc_vector = index_reader.get_document_vector(docid)
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

def rocchio_feedback(
    query_vector,
    rel_vectors,
    nrel_vectors=None,
    alpha=1.0,
    beta=0.75,
    gamma=0.0,
    top_fb_docs=10,
    top_fb_terms=128,
):
    query_vector = l2_normalize(query_vector)
    if top_fb_docs is not None:
        rel_vectors = rel_vectors[:top_fb_docs]
        if nrel_vectors:
            nrel_vectors = nrel_vectors[:top_fb_docs]

    mean_rel = compute_mean_vector(rel_vectors, fb_terms=top_fb_terms)
    mean_nrel = compute_mean_vector(nrel_vectors, fb_terms=top_fb_terms) if nrel_vectors else {}
    rocchio_vec = {}
    vocab = set(query_vector.keys()) | set(mean_rel.keys()) | set(mean_nrel.keys())
    for term in vocab:
        score = (
            alpha * query_vector.get(term, 0.0)
            + beta * mean_rel.get(term, 0.0)
            - gamma * mean_nrel.get(term, 0.0)
        )
        if score > 0:
            rocchio_vec[term] = score

    should = querybuilder.JBooleanClauseOccur["should"].value
    boolean_query_builder = querybuilder.get_boolean_query_builder()
    for term, weight in rocchio_vec.items():
        _term = JTermQuery(JTerm("contents", term))
        boost = querybuilder.get_boost_query(_term, weight)
        boolean_query_builder.add(boost, should)
    return boolean_query_builder.build()

