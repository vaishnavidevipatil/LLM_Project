import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

#function to get response from LLAma 2 Model

def getLLamaresponse(input_text, no_words, blog_style):
    
    # LLama Model created
    llm = CTransformers(model='C:\\Users\\vaish\\code\\LLM\\LLM_Project\\models\\llama-2-7b-chat.ggmlv3.q2_K.bin',
                        model_type='llama',
                        config={
                            'max_new_tokens': 256,
                            'temperature': 0.01
                        })
    
    # Prompt Template
    template = """
        Write a blog for {blog_style} about: {input_text}  
        Within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)

 
    # Generate the response from the LLaMA 2 model
    response = llm(prompt.format(blog_style=blog_style, 
                                  input_text=input_text,
                                  no_words=no_words))
   

    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 =st.columns([5,5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style=st.selectbox('Writing the blog for',('Researchers', 'DataScientists', 'Common People'), index=0)

submit = st.button("Generate")


#Final Response

if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))

"""
LLama 2 Chatbot – Streamlit App
--------------------------------
This Streamlit script turns your previous blog‑generator into an **interactive chatbot**.
Just run:
    streamlit run llama2_chatbot_streamlit.py

Key points
* Uses the *same* local ggml‑quantized LLama‑2 model via ``CTransformers``.
* Keeps full chat history inside ``st.session_state`` so the model can answer in context.
* Uses the modern Streamlit chat API (``st.chat_message`` + ``st.chat_input``) for a native look‑and‑feel.
* Model is loaded only once with ``@st.cache_resource`` so hot‑reloads are instant.
"""

# import streamlit as st
# # from langchain.llms import CTransformers
# from langchain_community.llms import CTransformers

# from typing import List, Dict

# # ──────────────────────────────────────────────────────────────────────────────
# # Helper – load the LLama 2 model ONCE and cache it across reruns
# # ──────────────────────────────────────────────────────────────────────────────

# @st.cache_resource(show_spinner="Loading LLama 2 ‑ this can take a minute…")
# def load_llama() -> CTransformers:
#     """Load the local quantised LLama 2 model with sane defaults."""
#     return CTransformers(
#         model=r"C:\\Users\\vaish\\code\\LLM\\LLM_Project\\models\\llama-2-7b-chat.ggmlv3.q2_K.bin",
#         model_type="llama",
#         config={
#             "max_new_tokens": 256,
#             "temperature": 0.7,
#             "context_length": 2048,
#         },
#     )

# llm = load_llama()

# # ──────────────────────────────────────────────────────────────────────────────
# # Streamlit UI setup
# # ──────────────────────────────────────────────────────────────────────────────

# st.set_page_config(
#     page_title="LLama 2 Chatbot",
#     page_icon="💬",
#     layout="centered",
#     initial_sidebar_state="collapsed",
# )

# st.title("LLama 2 Chatbot 💬")

# # Initialise chat history
# if "messages" not in st.session_state:
#     st.session_state.messages: List[Dict[str, str]] = []

# # Sidebar – controls
# with st.sidebar:
#     st.header("⚙️ Options")
#     temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7, 0.05)
#     max_tokens = st.slider("Max new tokens", 32, 512, 256, 32)
#     if st.button("🗑️ Clear chat"):
#         st.session_state.messages.clear()
#     # Apply sliders to the shared LLM config on‑the‑fly
#     llm.config["temperature"] = temperature
#     llm.config["max_new_tokens"] = max_tokens

# # Display previous messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # Chat input field (appears at bottom of page)
# user_prompt = st.chat_input("Type your message…")

# if user_prompt:
#     # 1️⃣ Echo user message to the chat log & UI
#     st.session_state.messages.append({"role": "user", "content": user_prompt})
#     with st.chat_message("user"):
#         st.markdown(user_prompt)

#     # 2️⃣ Build a conversation prompt for the model (simple UX, no system msgs)
#     conversation = "\n".join(
#         f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages
#     )
#     full_prompt = (
#         "You are a helpful AI assistant. Answer as concisely as appropriate.\n\n"
#         f"{conversation}\nAssistant:"
#     )

#     # 3️⃣ Generate assistant response (streaming disabled in CTransformers 0.2.x)
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking…"):
#             response = llm(full_prompt)
#             st.markdown(response.strip())

#     # 4️⃣ Save assistant response to history
#     st.session_state.messages.append({"role": "assistant", "content": response.strip()})
