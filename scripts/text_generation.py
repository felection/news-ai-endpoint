from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = "unsloth/gemma-3-1b-it"
# Explicitly set the `from_flax` parameter to False 
# to avoid loading a Flax tokenizer if it's present
tokenizer = AutoTokenizer.from_pretrained(model_name, from_flax=False)  
model = AutoModelForCausalLM.from_pretrained(model_name)

def rephrase_text(input_text):
    # A clearer prompt structure with explicit formatting to guide the model
    prompt = f"""
    <instruction>
    Rephrase the following text in a different style. Your response should ONLY contain the rephrased text, nothing else.
    </instruction>
    <input>
    {input_text}
    </input>

    <output>
    """
    # Encode the input text
    input = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
      # Generate the rephrased text
    outputs = model.generate(
        **input,
        num_beams=3,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        early_stopping=True       # 5. Enable early stopping
    )

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the rephrased part after the prompt
    # Find where the output section begins and extract text after it
    if "<output>" in generated_text:
        rephrased_text = generated_text.split("<output>")[1].strip()
    else:
        # If the model didn't respect the format, extract text after the input
        # This is a fallback approach
        parts = generated_text.split(input_text)
        if len(parts) > 1:
            rephrased_text = parts[1].strip()
        else:
            # Last resort: remove the prompt directly
            rephrased_text = generated_text.replace(prompt, "").strip()
    
    # Clean up any remaining tags or artifacts
    rephrased_text = rephrased_text.replace("</output>", "").strip()
    
    return rephrased_text


# Example usage
input_text = "The rapid expansion of the internet has revolutionized communication and information access globally"
rephrased = rephrase_text(input_text)
print("Rephrased Text:", rephrased)