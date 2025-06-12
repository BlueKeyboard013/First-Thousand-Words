import openai
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPEN_API_KEY")

# Sample words (or load your 500 most frequent ones)
words = [
    "travel", "transportation", "airport", "hotel", "directions", "tourism", "weather", "communication",
    "business", "office", "meetings", "email", "negotiation", "finance", "jobs",
    "school", "university", "subjects", "classroom", "studying", "homework",
    "food", "drinks", "restaurant", "cooking", "ingredients", "groceries", "menu",
    "body", "doctor", "medicine", "symptoms", "emergency", "hospital", "wellness",
    "clothing", "sizes", "money", "store", "bargaining", "electronics",
    "internet", "devices", "social media", "apps", "software", "coding", "AI",
    "family", "friends", "dating", "marriage", "emotions", "personal",
    "household", "chores", "morning", "evening", "routine", "weekend",
    "holidays", "traditions", "music", "art", "sports", "media",
    "animals", "plants", "landscape", "environment", "seasons"
]


# Get embeddings
# embedding is just a list of floats that represents the meaning of a text.
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# embeddings = [get_embedding(word) for word in words]

get_embedding("I am sick ")