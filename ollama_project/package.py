# import requests

# # Define the Ollama API endpoint
# OLLAMA_URL = "http://localhost:11434/api/generate"

# # Specify the model you want to use (e.g., 'llama3', 'mistral', etc.)
# model_name = "llama3"

# # Your prompt to the model
# prompt = "Explain the theory of relativity in simple words."

# # Make a request to the Ollama API
# response = requests.post(OLLAMA_URL, json={
#     "model": model_name,
#     "prompt": prompt,
#     "stream": False  # Set to True if you want to stream tokens
# })

# # Parse and print the response
# if response.status_code == 200:
#     result = response.json()
#     print("Generated Response:\n")
#     print(result.get("response"))
# else:
#     print("Failed to get a response from Ollama:", response.status_code)

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

model_name = "llama3"
prompt = "Tell me a joke about computers."

response = requests.post(OLLAMA_URL, json={
    "model": model_name,
    "prompt": prompt,
    "stream": False
})

if response.status_code == 200:
    result = response.json()
    print("Generated Response:\n")
    print(result.get("response"))
else:
    print("Failed to get a response from Ollama:", response.status_code)
    print("Error details:", response.text)
