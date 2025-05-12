from huggingface_hub import hf_hub_download
import time
from llama_cpp import Llama


# Download the GGUF model file
REPO_ID = "allenai/OLMo-2-0425-1B-Instruct-GGUF"
FILENAME = "OLMo-2-0425-1B-Instruct-Q4_K_M.gguf"  # 4-bit quantized version for fast CPU inference

# 2. Download GGUF model file
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

start_time = time.time()
# 3. Load the model with llama-cpp
llm = Llama(model_path=model_path, verbose=False)


# 4. Rephrasing function with Mistral-style prompt
def rephrase(text: str) -> str:
    prompt = f"""Rephrase the following sentence using different words but the same meaning. 
    Only return the rephrased sentence. 
    Text: "{text}" 
    Rephrased:"""

    result = llm(prompt, max_tokens=500, stop=["</s>", "\n"], echo=False)
    return result["choices"][0]["text"].strip()


def translate(text: str, output_language: str) -> str:
    prompt = f"""Translate the following sentence to {output_language}. Only return the translated sentence.
    Text: {text}
    Translation:"""
    result = llm(prompt, max_tokens=500, echo=False)
    return result["choices"][0]["text"].strip()


# 5. Example usage
input_text = "The team postponed the meeting because several members were unavailable."
output_text = rephrase(input_text)

# 6. Show results
print("🔹 Original:", input_text)
print("🔁 Rephrased:", output_text)
print("Processing time rephrase: ", time.time() - start_time)
time2 = time.time()
print("Translation to French:", translate(input_text, "French"))
print("Processing time Translation: ", time.time() - time2)
