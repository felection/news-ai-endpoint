"""
Emotion Detection using cardiffnlp/twitter-roberta-large-emotion-latest
---------------------------------------------------------------------
A simple script to identify emotions in text using the Twitter-RoBERTa model.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np


def detect_emotions(text):
    """
    Detect emotions in the provided text using cardiffnlp/twitter-roberta-large-emotion-latest model.

    Args:
        text (str): The text to analyze for emotions

    Returns:
        dict: Emotion labels and their corresponding scores
    """
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-large-emotion-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Prepare text inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        scores = scores.numpy()[0]

    # Get emotion labels
    emotion_labels = model.config.id2label

    # Create dictionary of emotions and scores
    emotions = {emotion_labels[i]: float(scores[i]) for i in range(len(scores))}

    # Sort emotions by score in descending order
    sorted_emotions = {
        k: v
        for k, v in sorted(emotions.items(), key=lambda item: item[1], reverse=True)
    }

    return sorted_emotions


def main():
    # Sample text
    text = "I'm so excited about the concert tonight! But I'm also a bit nervous about the crowd."

    print(f'\nAnalyzing text: "{text}"\n')

    # Detect emotions
    emotions = detect_emotions(text)

    # Display results
    print("Detected emotions (in descending order of confidence):")
    for emotion, score in emotions.items():
        print(f"- {emotion}: {score:.4f} ({score * 100:.1f}%)")

    # Get the dominant emotion
    dominant_emotion = next(iter(emotions))
    print(f"\nDominant emotion: {dominant_emotion}")


if __name__ == "__main__":
    main()
