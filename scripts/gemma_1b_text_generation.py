from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time

# Load the tokenizer and model
model_name = "unsloth/gemma-3-1b-it"
# Explicitly set the `from_flax` parameter to False
# to avoid loading a Flax tokenizer if it's present
tokenizer = AutoTokenizer.from_pretrained(model_name, from_flax=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=True,
    low_cpu_mem_usage=True,
)


def extract_text(text):
    # Try to find output content with proper closing tag first
    output_pattern = r"<output>(.*?)</output>"
    output_match = re.search(output_pattern, text, re.DOTALL)

    if output_match:
        output_content = output_match.group(1).strip()
    else:
        # If no closing tag, try to extract everything after <output>
        open_output_pattern = r"<output>(.*?)$"
        open_output_match = re.search(open_output_pattern, text, re.DOTALL)

        if open_output_match:
            output_content = open_output_match.group(1).strip()
        else:
            return "No <output> tag found"

    # Check for optional tags within the output content
    optional_tag_pattern = r"<([^>]+)>(.*?)</\1>"
    optional_match = re.search(optional_tag_pattern, output_content, re.DOTALL)

    if optional_match:
        # Optional tag exists, extract the text from it
        return optional_match.group(2).strip()
    else:
        # No optional tag, return the entire output content
        return output_content.strip()


def rephrase_text(input_text: str, instruction: str) -> str:
    # A clearer prompt structure with explicit formatting to guide the model
    prompt = f"""<instruction>
    {instruction}
    Do not add explanations or generate multiple versions. Show the rephrased text directly.
    </instruction>
    <input>
    {input_text}
    </input>
    <output>"""
    # Encode the input text
    input = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    # Generate the rephrased text
    outputs = model.generate(
        **input,
        max_new_tokens=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        # repetition_penalty=1.2,  # Penalize repetition more strongly
        # length_penalty=0.8,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        # early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,  # Proper padding
        # Force output to end with closing tag if possible
        eos_token_id=tokenizer.encode("</output>", add_special_tokens=False)[-1],
    )

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated_text = extract_text(generated_text)
    return generated_text


# Example usage
start_time = time.time()
input_text = """
Germany is dependent on imports for many critical raw materials, including Helium. The noble gas could also be obtained in Mecklenburg-West Pomerania. A company is looking on site. Helium knows many of them only from floating balloons, but the noble gas is also urgently needed in the medical field, in the production of computer chips or in research. In Germany it has not yet been promoted – but this could change. 45-8 Energy, based in the French butcher, is currently looking for Helium in Vorpommern. With the help of vibration vehicles and sensors, a kind of ultra-sound image is created that reaches up to 4000 meters in depth. The approval procedure shall be carried out for a similar project by the company in Brandenburg. The Helium used in this country and in Europe has so far mainly come from the USA, Qatar and Algeria. It is conventionally carried along with natural gas drilling and, during production, liquefied natural gas (LNG) is obtained from the conveyed gas mixture with high energy consumption. "As an industrial site, we are still 100% dependent on Helium," says Harald Kiefer of 45-8 Energy. "We want to change this here." According to Peter Klitzke from the Federal Institute for Geosciences and Raw Materials (BGR), the demand will continue to rise in the future – to seven to ten million cubic meters. The global market is about 170 million cubic meters. Over the next few years, global demand is expected to rise to more than 200 million cubic metres. "The price has gone up significantly in recent years," said Klitzke. Companies were increasingly interested in targeted funding. The project in Vorpommern has so far been unique in Germany, Klitzke said. If a subsidy is indeed possible economically, it could at least contribute to the supply of the North German market. 45-8 Energy knows about the Helium project thanks to data from GDR times. Hardly any country had been studied geophysically as well as the GDR, says geophysicist Andreas Schuck, who accompanied the project in Vorpommern. At the time, the GDR was mainly looking for natural gas or oil. Between Greifswald and Wolgast, deep bores instead had a gas mixture of nitrogen and helium. At that time, however, the helium has not been used for a lack of technical application, says Schuck. The measurements started at the end of January and are scheduled to be completed next week. The company expects a comprehensive result in a half to a year. Among other things, it is sought with approximately 27 tons of heavy and 10 meters long exploration vehicles. © Patrick Mariathasan / THE SPIEGEL © Patrick Mariathasan / THE SPIEGEL
"""
instruction = """
Rephrase the following text in a different way while keeping the same meaning and without touching numbers. Do not rephrase names and dates and numbers and other static elements.
IMPORTANT: ALL numbers, measurements, percentages, and dates MUST remain EXACTLY as they appear in the original text. For example, if the original has '45-8 Energy', '100%', '4000 meters', or '170 million cubic meters', these MUST appear unchanged.
"""
rephrased = rephrase_text(input_text, instruction)

process_time = time.time() - start_time
print("Processing Time:", process_time)
print("Rephrased Text:", rephrased)
