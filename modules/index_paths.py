from pyserini.search import get_topics, get_qrels

THE_SPARSE_INDEX = {
    'dl19': 'msmarco-v1-passage-full',
    'dl20': 'msmarco-v1-passage-full',
    'dl21': 'msmarco-v2-passage-full',
    'dl22': 'msmarco-v2-passage-full',
    'dl23': 'msmarco-v2-passage-full',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat', 
    'nfcorpus': 'beir-v1.0.0-nfcorpus.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    }

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'dl21': 'dl21',
    'dl22': 'dl22',
    'dl23': 'dl23',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'nfcorpus': 'beir-v1.0.0-nfcorpus-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'arguana': 'beir-v1.0.0-arguana-test',
}

def load_queries_qids(corpus_name):
    topics = get_topics(THE_TOPICS[corpus_name] if corpus_name != 'dl20' else 'dl20')
    if corpus_name in ['dl21', 'dl22', 'dl23']:
        qrels = get_qrels(f'{THE_TOPICS[corpus_name]}-passage')
    else:
        qrels = get_qrels(THE_TOPICS[corpus_name])
    test_only_qids_queries = set(qrels.keys())
    topics_qids = [(key, topics[key]['title'])  for key in topics if key in test_only_qids_queries]
    qids = [i[0] for i in topics_qids]
    queries = [i[1] for i in topics_qids]
    return qids, queries

