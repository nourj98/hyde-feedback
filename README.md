# Feedback Models for HyDE

This is the code for the paper: [Revisiting Feedback Models for HyDE](https://arxiv.org/abs/2511.19349)

## Overview

Recent approaches that leverage large language models (LLMs) for pseudo-relevance feedback (PRF) have generally not utilized well-established feedback models like Rocchio and RM3 when expanding queries for sparse retrievers like BM25. Instead, they often opt for a simple string concatenation of the query and LLM-generated expansion content. But is this optimal? To answer this question, we revisit and systematically evaluate traditional feedback models in the context of HyDE, a popular method that enriches query representations with LLM-generated hypothetical answer documents. Our experiments show that HyDE's effectiveness can be substantially improved when leveraging feedback algorithms such as Rocchio to extract and weight expansion terms, providing a simple way to further enhance the accuracy of LLM-based PRF methods.


## Setup

We use the following packages:

```
vllm==0.10.1.1
pyserini==1.2.0
transformers==4.56.1
torch==2.7.1
```

## Run HyDE with feedback models!
To run HyDE with your favorite feedback method, run the following command:

```
python run_hyde.py \
    --model_path $model \
    --corpus_name $corpus \
    --returned_hits 100 \
    --feedback_mechanism $feedback_mechanism \
    --output_folder $output_folder \
```

where:

```--model_path```: the path to your LLM (Code works with Qwen2.5-7B-Instruct, Qwen3-14B, gpt-oss-20b; minor adaptations may be needed for other LLMs) \
```--corpus_name```: Corpus to evaluate on (e.g., 'dl19', 'dl20', 'news', 'covid', 'fiqa', etc; See modules/index_paths.py) \
```--returned_hits```: Number of passages to retrieve with HyDE \
```--feedback_mechanism```: Feedback mechanism for creating HyDE query (Options: 'hyde', 'rocchio', 'rm3', 'mugi', 'naive') \
```--output_folder```: Folder to save HyDE ranking 

This general code should extend to additional corpora (as long as its in Pyserini). To do so, just add to THE_SPARSE_INDEX and THE_TOPICS in ```modules/index_paths.py``` and add a HyDE prompt to ```modules/hyde.py```. 

Running our Query2Doc and generic BM25 pseudo-relevance feedback  baselines should be just as simple as the above!

Query2Doc:
```
python run_query2doc.py \
    --model_path $model \
    --corpus_name $corpus \
    --returned_hits 100 \
    --output_folder $output_folder \
```

This will run our baseline which generates a single hypothetical passage and repeats the query 5 times.

And, for traditional PRF with BM25:
```
python run_prf.py \
    --corpus_name $corpus \
    --feedback_documents 8 \
    --returned_hits 100 \
    --feedback_mechanism $feedback_mechanism  \
    --output_folder $output_folder \
```

If you leave ```--feedback_mechanism``` blank, this will run standard BM25. Otherwise, this will run similarly to ```run_hyde.py```

## Citation

Please cite our paper if it is helpful to your work!
```
@article{jedidi2025revisiting,
  title={Revisiting Feedback Models for HyDE},
  author={Jedidi, Nour and Lin, Jimmy},
  journal={arXiv preprint arXiv:2511.19349},
  year={2025}
}
```
