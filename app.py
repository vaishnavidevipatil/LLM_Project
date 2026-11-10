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
