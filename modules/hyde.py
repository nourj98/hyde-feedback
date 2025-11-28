import os
from transformers import AutoTokenizer 

import torch
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


WEB_SEARCH = """Please write a passage to answer the question.
Question: {}
Passage:"""


SCIFACT = """Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:"""


ARGUANA = """Please write a counter argument for the passage.
Passage: {}
Counter Argument:"""


TREC_COVID = """Please write a scientific paper passage to answer the question.
Question: {}
Passage:"""


FIQA = """Please write a financial article passage to answer the question.
Question: {}
Passage:"""


DBPEDIA_ENTITY = """Please write a passage to answer the question.
Question: {}
Passage:"""


TREC_NEWS = """Please write a news passage about the topic.
Topic: {}
Passage:"""

SCIDOCS = """Please write a scientific paper passage that would be relevant to the following paper abstract.
Title: {}
Related Passage:"""

NQ = """Please write a Wikipedia article to answer the question.
Question: {}
Passage:"""


prompt_dict = {
    # Web-style collections
    'dl19': WEB_SEARCH,
    'dl20': WEB_SEARCH,
    'dl21': WEB_SEARCH,
    'dl22': WEB_SEARCH,
    'dl23': WEB_SEARCH,
    'robust04': TREC_NEWS,
    'dbpedia': DBPEDIA_ENTITY,

    # Scientific and fact-checking collections
    'scifact': SCIFACT,
    'covid': TREC_COVID,
    'scidocs': SCIDOCS,

    # QA datasets
    'nq': NQ,
    'hotpotqa': NQ,

    # Financial and argumentation datasets
    'fiqa': FIQA,
    'arguana': ARGUANA,

    # News datasets
    'news': TREC_NEWS,

    # Biomedical dataset
    'nfcorpus': TREC_COVID,
}

class HyDE():
    def __init__(
        self,
        base_model_name_or_path: str,
        batch_size: int = 999999999999,
        context_size: int = 32000,
        max_output_tokens: int = 8192,
        fp_options: str = "float16",
        num_gpus: int = 1,
        device: str = "cuda",
        dataset_prompt: str = 'default',
    ):

        self.model_name = base_model_name_or_path
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.num_gpus = num_gpus
        self.device = device
        self.dataset_prompt = dataset_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
           # dtype=fp_options,
            dtype='bfloat16',
            gpu_memory_utilization=0.9,
            enforce_eager=True,
        )
    
    def _generate_model_outputs(self, prompts):
        return self.model.generate(prompts, self.sampling_params) 
    
    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
    
    def _extract_gptoss_content(self, text):
        passage_text = text.split("<|start|>assistant<|channel|>final<|message|>")[-1].strip()
        passage_text = self.truncate(passage_text, 512)
        return passage_text

    def _process_with_vllm_hyp_documents(self, prompts):
        outputs = self._generate_model_outputs(prompts)  
        all_passages = []
        for i, output in enumerate(outputs):
            sample_passages = [] 
            for sample in output.outputs:
                text = sample.text
                if 'gpt-oss' in self.model_name:
                    text = self._extract_gptoss_content(text)
                sample_passages.append(text)
            all_passages.append(sample_passages)
        return all_passages

    def return_prompt_hyde(self, query, task) -> str:
        chat = [{'role': "user", 'content': prompt_dict[task].format(query)}]
        prompt_text = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return prompt_text
    
    @torch.inference_mode()
    def predict(self, queries, task):
        prompts = [self.return_prompt_hyde(query, task) for query in queries]  
        print(prompts[0])
        if 'gpt-oss' in self.model_name:
            # I believe reasoning tokens get included, so we will do
            # something hacky: increase max tokens then truncate...
            self.sampling_params = SamplingParams(temperature=0.7,
                                                  max_tokens=1024,
                                                  skip_special_tokens=False,
                                                  n=8)
        else:
            self.sampling_params = SamplingParams(temperature=0.7,
                                                  max_tokens=512,
                                                  skip_special_tokens=False,
                                                  n=8)
        outputs = self._process_with_vllm_hyp_documents(prompts)
        return outputs

class Query2Doc(HyDE):
    def _extract_gptoss_content(self, text):
        passage_text = text.split("<|start|>assistant<|channel|>final<|message|>")[-1].strip()
        passage_text = self.truncate(passage_text, 128)
        return passage_text

    def _process_with_vllm_hyp_documents(self, prompts):
        outputs = self._generate_model_outputs(prompts)  
        all_passages = []
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            if 'gpt-oss' in self.model_name:
                text = self._extract_gptoss_content(text)
            all_passages.append(text)
        return all_passages

    @torch.inference_mode()
    def predict(self, queries, task):
        prompts = [self.return_prompt_hyde(query, task) for query in queries]  
        print(prompts[0])
        if 'gpt-oss' in self.model_name:
            # I believe reasoning tokens get included, so we will do
            # something hacky: increase max tokens then truncate...
            self.sampling_params = SamplingParams(temperature=0,
                                                  max_tokens=512,
                                                  skip_special_tokens=False)
        else:
            self.sampling_params = SamplingParams(temperature=0,
                                                  max_tokens=128,
                                                  skip_special_tokens=False)
        outputs = self._process_with_vllm_hyp_documents(prompts)
        return outputs