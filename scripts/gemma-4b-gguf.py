# gemma-3-4b-it-Q6_K.gguf
# unsloth/gemma-3-4b-it-GGUF Medium
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import time

# 1. Model info
REPO_ID = "unsloth/gemma-3-4b-it-GGUF"
FILENAME = "gemma-3-4b-it-Q6_K.gguf"

# 2. Download GGUF model file
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

start_time = time.time()
# 3. Load the model with llama-cpp
llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)


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
    print(result)
    return result["choices"][0]["text"].strip()


# 5. Example usage
input_text = """Dem ZDF schlägt eine Welle der Empörung entgegen: In einer Wahl-Sendung haben die Veranstalter offenbar bewusst die Stimmung beeinflusst, indem das Publikum einseitig zugunsten von Linken und Grünen eingeladen wurde. Dieser Verdacht steht nach der Ausstrahlung der Sendung „ Schlagabtausch “ am Donnerstag im Raum! Ein Mitarbeiter des Senders hat die politisch eindimensionale Zusammensetzung des Publikums sogar schon zugegeben! Jetzt kommen in BILD die Gäste der Sendung zu Wort. Sie bekamen keinen Applaus – Lindner, Dobrindt, Wagenknecht und Chrupalla ▶︎ CSU -Landesgruppenchef Alexander Dobrindt (54) nimmt es gelassen und antwortet ironisch gegenüber BILD: „Ist uns allen schon mal passiert. Schwups, in der Zeile verrutscht und schon hat man die Gästeliste der ,heute-show‘ eingeladen.“ ▶︎ FDP -Chef Christian Lindner (46) gesteht gegenüber BILD: „In der Sendung konnte ich einen Lachanfall kaum unterdrücken.“ Er warnt aber: „In Wahrheit ist es natürlich eine ernste Sache, wenn gebührenfinanzierte Sendungen als einseitig wahrgenommen werden. Das verschärft die wachsenden Vorbehalte in der Bevölkerung.“ ▶︎ BSW -Chefin Sahra Wagenknecht (55) vermutet gegenüber BILD ein Kalkül des Senders: „Wir merken schon, man versucht natürlich, uns mit all dem kleinzukriegen, weil wir stören, weil natürlich die alten Parteien nicht wollen, dass wir tatsächlich auch im nächsten Bundestag mit einer starken Fraktion vertreten sind.“ Der Grund laut Wagenknecht: „Vor uns haben sie eben Angst, weil wir sie unter Druck setzen, ihre Politik nicht teilen.“ ▶︎ AfD -Co-Vorsitzender Tino Chrupalla (49) sagt auf BILD-Anfrage: „Mit diesem Setting hat der öffentlich-rechtliche Rundfunk eine Werbesendung für Linke und Grüne veranstaltet.“ Chrupalla prangert an: „Diese Manipulation der Wähler verstößt gegen die Gebote der Unparteilichkeit und Ausgewogenheit und bestätigt tatsächlich den Vorwurf, ARD und ZDF verfolgten eine linksgrüne Agenda.“ Sie ernteten großen Applaus – Jan van Aken und Felix Banaszak ▶︎ Den Linke -Co-Vorsitzenden Jan van Aken (63) scheint das ihm wohlgesonnene, einseitige Publikum nicht gestört zu haben. Van Aken zu BILD: „Zur Auswahl der Gäste kann ich nichts sagen, außer dass wir niemanden dahin mobilisiert haben. Kein Mensch wird zum Beifall gezwungen, demnach freue ich mich einfach, dass meine Positionen zu Mieten und Preisen viel Anklang gefunden haben.“ ▶︎ Grünen -Co-Chef Felix Banaszak (35) antwortete auf BILD-Anfrage nicht."""
output_text = rephrase(input_text)

# 6. Show results
print("🔹 Original:", input_text)
print("🔁 Rephrased:", output_text)
print("Processing time rephrase: ", time.time() - start_time)
""" time2 = time.time()
print("Translation to French:", translate(input_text, "French"))
print("Processing time Translation: ", time.time() - time2)
 """
