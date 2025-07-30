from transformers import pipeline

# Load the model and set device to GPU (device=0)
model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    device=0  # Use GPU (0 refers to the first GPU)
)

# Generate text
output = model("What is LangChain?")
print(output)