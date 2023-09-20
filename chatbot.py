import streamlit as st
import torch
from typing import Any, Dict, Optional, Tuple
from dataclasses import asdict, dataclass, field

from transformers.generation.utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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

@st.cache_resource
def init_model():

    generating_args = GeneratingArguments
    model, tokenizer = load_pretrained()
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name="GanymedeNil/text2vec-large-chinese",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.load_local("faiss_index", embeddings)

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
    
    return model, tokenizer, db


# App title
st.set_page_config(page_title="QA_system Chatbot")
st.title("Retrieval-Augmented LLMs")

def main():
    llm, tokenizer, db = init_model()
    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": ""}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Function for generating BaiChuan response
    def generate_baichuan_response(prompt):
        # output = replicate.run(llm, 
        #                        input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
        #                               "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
        # st.session_state.messages.append({"role": "user", "content": prompt})
        messages = st.session_state.messages
        for response in llm.chat(tokenizer, messages, stream=True):
            # print(response)
            yield response

    def generate_baichuan_prompt(prompt):
        docs=db.similarity_search(prompt)
        question = docs[0].page_content
        answer = docs[0].metadata["answer"]
        TEMPLATE = """已知{question}\n{answer}\n{prompt}"""
        return TEMPLATE.format(question, answer, prompt)
    

    # User-provided prompt
    if prompt := st.chat_input(disabled=False):
        prompt = generate_baichuan_prompt(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                placeholder = st.empty()
                for res in generate_baichuan_response(prompt):
                    placeholder.markdown(res)

        message = {"role": "assistant", "content": res}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
    # quick start
    # `streamlit run chatbot.py`