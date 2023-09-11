import streamlit as st
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

@st.cache_resource
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


# App title
st.set_page_config(page_title="QA_system Chatbot")
st.title("QA_System")

def main():
    llm, tokenizer = init_model()

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
        return prompt
    

    # User-provided prompt
    if prompt := st.chat_input(disabled=False):
        prompt = generate_baichuan_prompt(prompt)
    #     prompt = """{"question": "WiFi调制技术？带宽？遵循的协议？连接的方式？", "answer": "Wi-Fi调制技术是一种用于实现无线局域网络（WLAN）的调制技术，主要用于在无线信道中传输数据。Wi-Fi调制技术采用了OFDM（正交频分复用）和DSSS（直接序列扩频）等多种调制技术，可以实现高速、可靠的数据传输。Wi-Fi标准规定了不同频段和带宽的无线网络的工作方式，其中最常用的是2.4GHz和5GHz频段的Wi-Fi网络。Wi-Fi网络的带宽通常为20MHz、40MHz、80MHz或160MHz，不同带宽的网络可以支持不同的数据传输速率。在协议方面，Wi-Fi网络遵循了IEEE 802.11系列标准，其中包括了802.11a、802.11b、802.11g、802.11n、802.11ac、802.11ax等多个版本。不同版本的协议支持的数据传输速率和信道带宽等不同，可以根据具体的应用需求进行选择。在连接方式方面，Wi-Fi网络通常采用基于无线接入点（Access Point，AP）的无线连接方式，通过无线接入点来连接多个无线设备，形成一个无线局域网络。此外，也可以采用无线点对点或者无线Mesh网络等方式进行连接。Wi-Fi调制技术是一种重要的无线通信技术，具有高速、可靠、灵活等优点，在移动互联网应用、智能家居、工业自动化等领域得到广泛应用。"}, {"question": "通信原理：3GPP", "answer": "3GPP（Third Generation Partnership Project）是一个由全球电信标准化组织联盟组成的合作机构，成立于1998年。3GPP致力于制定和发布全球通信标准，包括移动通信、移动宽带和多媒体通信等领域。它是制定移动通信技术标准的主要组织之一。3GPP的成员包括来自全球各地的电信运营商、电信设备制造商、研究机构和其他相关组织。3GPP的成员共同合作，制定新的技术规范和标准，以确保各种通信设备和系统之间的互操作性和相互兼容性。3GPP的工作是开放的，它允许任何人或组织参与，并通过开放式讨论、测试和评估来制定标准。3GPP制定的标准主要包括GSM、UMTS、LTE、5G等移动通信技术标准。这些标准是全球通信行业的基础，使得不同制造商的设备可以相互兼容和交互操作，从而实现了全球范围内的移动通信互联互通。同时，这些标准也推动了移动通信技术的快速发展和普及，为用户提供了更加便捷和高效的通信服务。"}
    # 介绍一下wifi调制技术？"""
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
    # `streamlit run chatbot`