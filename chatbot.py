import streamlit as st
import torch
import os
from streamlit_option_menu import option_menu
from dialogue.dialogue import dialogue_page

# App title
# st.set_page_config(page_title="QA_system Chatbot")
# st.title("Retrieval-Augmented LLMs")

# def main():
    # llm, tokenizer, embeddings = init_model()
    # vectorstore = vs(embeddings)
    # Store LLM generated responses
    # if "messages" not in st.session_state.keys():
    #     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.write(message["content"])

    
    # def clear_chat_history():
    #     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
    # st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    
    # Function for generating BaiChuan response
    # def generate_baichuan_response():
    #     # output = replicate.run(llm, 
    #     #                        input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
    #     #                               "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    #     # st.session_state.messages.append({"role": "user", "content": prompt})
    #     messages = st.session_state.messages
    #     for response in llm.chat(tokenizer, messages, stream=True):
    #         yield response

    # def generate_baichuan_prompt(prompt):
    #     qas = vectorstore.do_search(prompt, topk=1)
    #     question = qas[0]["question"]
    #     answer = qas[0]["answer"]
    #     TEMPLATE = """已知{question}\n{answer}\n{prompt}"""
    #     return TEMPLATE.replace("{question}", question).replace("{answer}", answer).replace("{prompt}", prompt)
    
    # # User-provided prompt
    # if prompt := st.chat_input(disabled=False):
    #     with st.chat_message("user"):
    #         st.write(prompt)
    #     prompt = generate_baichuan_prompt(prompt)
    #     st.session_state.messages.append({"role": "user", "content": prompt})
        
    
    # # Generate a new response if last message is not from assistant
    # if st.session_state.messages[-1]["role"] != "assistant":
    #     with st.chat_message("assistant"):
    #         with st.spinner("Thinking..."):
    #             placeholder = st.empty()
    #             for res in generate_baichuan_response():
    #                 placeholder.markdown(res)

    #     message = {"role": "assistant", "content": res}
    #     st.session_state.messages.append(message)

if __name__ == "__main__":
    st.set_page_config(
        "CS chatbot system",
        os.path.join("img", "avtor.png"),
        initial_sidebar_state="auto",
    )
    st.title("欢迎使用[`CS ChatBOT`](https://github.com/iGangao/QAsystem) !")
    st.toast(
        f"欢迎使用 [`CS ChatBOT`](https://github.com/iGangao/QAsystem) ! \n\n"
    )
    with st.chat_message("assistant"):
        st.write("请问有什么我可以帮助你的吗？")
    pages = {
        "对话": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "知识库管理": {
            "icon": "hdd-stack",
            "func": dialogue_page,
        },
        "模型配置": {
            "icon": "hdd-stack",
            "func": dialogue_page,
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "avtor.png"
            ),
            use_column_width=True
        )
        st.caption(
            f"""<p align="right">当前版本：0.1</p>""",
            unsafe_allow_html=True,
        )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"]()
    # quick start
    # `streamlit run chatbot.py`