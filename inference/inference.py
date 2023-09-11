import torch
from typing import Any, Dict, Optional, Tuple
from dataclasses import asdict, dataclass, field

from transformers.generation.utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

@dataclass
class GeneratingArguments:
    """
    Arguments pertaining to specify the decoding parameters.
    """
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."}
    )
    temperature: Optional[float] = field(
        default=0.3,
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_p: Optional[float] = field(
        default=0.85,
        metadata={"help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."}
    )
    top_k: Optional[int] = field(
        default=5,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."}
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."}
    )
    repetition_penalty: Optional[float] = field(
        default=1.1,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    length_penalty: Optional[float] = field(
        default=2.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."}
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", None):
            args.pop("max_length", None)
        return args

def load_pretrained() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        trust_remote_code=True,
        cache_dir = None,
        revision = "main",
        use_auth_token = None,
        user_fast = False,
        padding_side="right",
    )
    # Load and prepare pretrained models (without valuehead).
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        trust_remote_code=True,
        cache_dir = None,
        revision="main",
        use_auth_token=None,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir = None,
        revision="main",
        use_auth_token=None,
    )

    # # Register auto class to save the custom code files.
    if hasattr(config, "auto_map") and "AutoConfig" in config.auto_map:
        config.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
        tokenizer.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoModelForCausalLM" in config.auto_map:
        model.__class__.register_for_auto_class()

    return model, tokenizer

def init_model():

    generating_args = GeneratingArguments
    model, tokenizer = load_pretrained()

    model.generation_config = GenerationConfig.from_pretrained(
        "baichuan-inc/Baichuan-13B-Chat",
    )
    model.generation_config.max_new_tokens = generating_args.max_new_tokens
    model.generation_config.temperature = generating_args.temperature
    model.generation_config.top_k = generating_args.top_k
    model.generation_config.top_p = generating_args.top_p
    model.generation_config.repetition_penalty = generating_args.repetition_penalty
    model.generation_config.do_sample = generating_args.do_sample
    model.generation_config.num_beams = generating_args.num_beams
    model.generation_config.length_penalty = generating_args.length_penalty
    
    return model, tokenizer

def inference():
    llm, tokenizer = init_model()

    import json
    qas = []
    with open("./all_question.json", "r") as jf:
        lines = jf.readlines()
        for line in lines:
            qas.append(json.loads(line))
    

    for qa in qas:
        messages = []
        messages.append({"role": "user", "content": qa['Q']})
        output = llm.chat(tokenizer, messages)
        data={
            "Q":qa["Q"],
            "output":output
        }
        with open("./only_inference.json", "a") as file:
            file.write(json.dumps(data, ensure_ascii=False,) + "\n")
            data = {}

def inference_based_retrieval():
    llm, tokenizer = init_model()

    import json
    qas = []
    with open("./all_question.json", "r") as jf:
        lines = jf.readlines()
        for line in lines:
            qas.append(json.loads(line))
    from tqdm import tqdm
    for qa in tqdm(qas, desc="Processing",mininterval=0.1):
        messages = []
        messages.append({"role": "user", "content": qa['Q']})
        messages.append({"role": "assistant", "content": qa["A"]})
        messages.append({"role": "user", "content": qa['Q']})
        output = llm.chat(tokenizer, messages)
        data={
            "Q":qa["Q"],
            "A":qa["A"],
            "output":output
        }
        with open("./retrieval_inference.json", "a") as file:
            file.write(json.dumps(data, ensure_ascii=False,) + "\n")
            data = {}

if __name__ == "__main__":
    inference()
    # inference_based_retrieval()