# from langchain.chat_models import ChatOpenAI
# from langchain.schema import HumanMessage
# import os
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("OPENAPI_KEY")

# print(api_key)
# # Initialize the chat model
# chat = ChatOpenAI(openai_api_key=api_key, temperature=0.7)

# # Basic loop
# print("Chatbot: Hello! Ask me anything. Type 'exit' to stop.")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'exit':
#         print("Chatbot: Goodbye!")
#         break
#     response = chat([HumanMessage(content=user_input)])
#     print("Chatbot:", response.content)

import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the model from Hugging Face Hub
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # You can change this model
    model_kwargs={"temperature": 0.7, "max_length": 200},
    huggingfacehub_api_token=hf_token
)

print("Chatbot: Hello! Ask me anything. Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = llm.invoke(user_input)
    print("Chatbot:", response)
