"""
Named Entity Recognition (NER) script using Hugging Face transformers.
This script processes text to identify and categorize entities.
"""

import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class NamedEntityRecognizer:
    """Class for performing named entity recognition on text."""
    
    # Entity type mapping from model tags to readable categories
    ENTITY_TYPE_MAP = {
        'B-MISC': 'Miscellaneous',
        'I-MISC': 'Miscellaneous',
        'B-PER': 'Person',
        'I-PER': 'Person',
        'B-ORG': 'Organization',
        'I-ORG': 'Organization',
        'B-LOC': 'Location',
        'I-LOC': 'Location'
    }
    
    # Default entity categories
    DEFAULT_CATEGORIES = {
        "Person": [],
        "Organization": [],
        "Location": [],
        "Miscellaneous": [],
        "Date": [],
        "Money": []
    }
    
    def __init__(self, model_name="dslim/bert-base-NER", min_score=0.80):
        """
        Initialize the NER processor.
        
        Args:
            model_name (str): Name of the HuggingFace model to use
            min_score (float): Minimum confidence score threshold for entities
        """
        self.min_score = min_score
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
    
    def process_text(self, text):
        """
        Process text to identify named entities.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary of categorized entities
        """
        # Get raw entities from the model
        raw_entities = self.nlp(text)
        
        # Process and group entities
        grouped_entities = self._group_entities(raw_entities)
        
        # Clean up the results
        return self._cleanup_entities(grouped_entities)
    
    def _group_entities(self, entities):
        """
        Group raw entity tokens into complete named entities.
        
        Args:
            entities (list): Raw entity tokens from the model
            
        Returns:
            dict: Grouped entities by category
        """
        grouped_entities = self.DEFAULT_CATEGORIES.copy()
        current_entity = {"text": "", "type": None, "score": 0.0}
        
        for entity in entities:
            word = entity["word"]
            entity_tag = entity["entity"]
            label = self.ENTITY_TYPE_MAP.get(entity_tag, "Other")
            score = float(entity["score"])
            is_subword = word.startswith("##")
            
            # Handle new entity (B- prefix)
            if entity_tag.startswith("B-"):
                # Save previous entity if it exists
                if current_entity["type"]:
                    current_entity["text"] = current_entity["text"].strip()
                    grouped_entities[current_entity["type"]].append(current_entity)
                
                # Start new entity
                current_entity = {
                    "text": word,
                    "type": label,
                    "score": score,
                }
            
            # Handle continuation of entity (I- prefix)
            elif entity_tag.startswith("I-") and current_entity["type"] == label:
                if is_subword:
                    # Append without space
                    current_entity["text"] += word.replace("##", "")
                else:
                    # New full word -> add space
                    current_entity["text"] += " " + word
                # Keep tracking the score
                current_entity["score"] = max(current_entity["score"], score)
            
            # Handle other cases (entity type change or non-entity)
            else:
                if current_entity["type"]:
                    current_entity["text"] = current_entity["text"].strip()
                    grouped_entities[current_entity["type"]].append(current_entity)
                current_entity = {"text": "", "type": None, "score": 0.0}
        
        # Add the final entity if one exists
        if current_entity["type"]:
            current_entity["text"] = current_entity["text"].strip()
            grouped_entities[current_entity["type"]].append(current_entity)
            
        return grouped_entities
    
    def _cleanup_entities(self, entity_dict):
        """
        Clean up entities by removing duplicates and low-confidence entities.
        
        Args:
            entity_dict (dict): Dictionary of categorized entities
            
        Returns:
            dict: Cleaned dictionary of entities
        """
        for category, entities in entity_dict.items():
            if not entities:  # Skip empty lists
                continue
                
            # Remove duplicates while preserving order
            seen_texts = set()
            unique_entities = []
            
            for entity in entities:
                # Skip low-quality entities
                if (entity['text'].startswith('##') or 
                    len(entity['text']) <= 1 or 
                    entity['score'] < self.min_score):
                    continue
                    
                if entity['text'] not in seen_texts:
                    seen_texts.add(entity['text'])
                    unique_entities.append(entity)
            
            entity_dict[category] = unique_entities
        
        return entity_dict


def main():
    """Main function to demonstrate the NER functionality."""
    # Example text
    example = """
    Namibia's founder president is dead. Sam Nujoma died on Saturday at the age of 95. Namibia's President Nangolo Mbumba announced this in national broadcasting and on Facebook. Nujoma has been the last "founding father" of African states. For decades, the former railway worker fought for his ideal of a free country in which all people have equal rights and opportunities. His desire for the "Joch of colonial oppression" forced him into exile for almost 30 years, from where the full-time liberty hero politically and militaryly organized the resistance against the occupation power of the racist South African apartheid regime. After independence in 1990, he became the first democratically elected President of Namibia – an office he held until 2005. He was a yovial state steer, whose greatest achievements included the policy of reconciliation among blacks and whites, which enabled a smooth transition to independence. "We were denied the simplest and most basic human rights of self-determination and independence," Nujoma said at the end of 2017 at the Party Day of the Government Party Swapo, which he co-founded. "We are proud to recall that we fought Namibians with strength and determination to free ourselves from the chains of colonial oppression and apartheid colonialism," Nujoma continued. Namibia was a German colony (1884-1915), then still called German-South West Africa for three decades. Then the great neighbour South Africa took power. No undisputed freedom fighter Nujoma was not an undisputed freedom fighter: Unlike South Africa's Nelson Mandela, he didn't sit in prison for nearly three decades, but he was also responsible for the shadows of the partially brutal liberation struggle. Internal critics should have persecuted, tortured or imprisoned the Swapo leadership. Hundreds of children have been dragged into training camps in neighbouring Angola and abused as child soldiers. Samuel Shafishuna Nujoma became 12. May 1929 was born first of eleven children of a peasant family of the Ovambo-Volkes in the north of the country. "Like all the boys at the time, I had to keep the cattle of my parents and help them in the house and in agriculture," he said in May 2018 on the occasion of his 89. Birthday. Later, he continued with evening courses and became active in the trade union before being elected as Chairman of a precursor organisation of the Swapo in 1959. At the end of the year, he was arrested for a protest. On March 1, 1960 he fled to exile where he co-founded the Swapo (South West Africa People's Organization). In the following years he traveled around the world to promote the independence of his home country. In 1966, the United Nations General Assembly placed Namibia under UN administration. The South African occupiers, however, do not know what the Swapo called the armed struggle. Only in 1989, when the apartheid regime began to roar at home, the South Africans departed. In September Nujoma returned to Namibia. Friendly relations with Cuba and North Korea "The fate of this country is now entirely in our hands," he said about six months later when he was appointed the first president of independent Namibia. Nujoma took radical reforms and began to revive the country. At the end of the 1990s, however, corruption accusations increased against the two-thirds-governing Swapo. In 1999, Parliament amended the Constitution to allow Nujoma a third term. His friendly relations with North Korea and Cuba were irritating some western donor countries. The relations with the former colonial power of Germany were also good, Nujoma called the Germans "removed cousins". Berlin became one of the most important donors for development projects in the country. Claims for a financial reparation of the smaller Herero and Nama tribes who suffered enormously under the brutal German colonial rule did not support his government dominated by the Ovambo tribe. Conflict, however, kept the issue of expropriations, because most of the land was in the hands of white farmers. All soil in Namibia belongs to the Namibian people, so the government can also expropriate, Nujoma explained in an interview with the "world" in 2002. The country "was overthrown by German colonialists from 1884 to 1915, they have divided our country among themselves, they have never acquired the ground and ground," he said. Despite the sabering, the land reform was mostly progressing in law and dragging. Critics who quickly wanted to expropriate all the whites, he kept in his mind. Namibia has become one of Africa's most stable democracies. At the age, his public appearances became more rare. He last had a few months after his 90th birthday on 12. May 2019 his deceased companion, Simbabwe's former president Robert Mugabe, proved the last honour at a funeral in Harare.
    """

    # Create NER processor and analyze text
    ner = NamedEntityRecognizer()
    results = ner.process_text(example)
    
    # Print results
    print("Named Entities Found:",results)

if __name__ == "__main__":
    main()