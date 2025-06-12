import os
from dotenv import load_dotenv
import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import word_tags
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


DATA_PATH = "data/"
CHROMA_PATH = "chroma"
load_dotenv()
API_KEY = os.getenv("OPEN_API_KEY")


# Example 500 words (you'd replace this with your real dataset)
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
]  # Simulate 500 words

# words = words[:500]  # Limit to exactly 500

# Define sample tag clusters with representative anchor words
tag_clusters = {
    "travel": ["travel", "airport", "luggage", "ticket", "itinerary"],
    "health": ["doctor", "hospital", "medicine", "emergency", "wellness"],
    "food": ["food", "restaurant", "menu", "cooking", "groceries"],
    "technology": ["devices", "internet", "AI", "software", "apps"],
    "education": ["school", "university", "studying", "homework", "classroom"],
    "business": ["office", "email", "negotiation", "finance", "jobs"],
    "daily life": ["morning", "evening", "routine", "chores", "weekend"],
    "relationships": ["family", "friends", "marriage", "dating", "emotions"],
    "arts and culture": ["music", "art", "media", "holidays", "traditions"],
    "environment": ["plants", "animals", "landscape", "environment", "seasons"]
}

# This is a placeholder function — we would use OpenAI API or HuggingFace to fetch real embeddings.
def dummy_embed(text):
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1536)  # Simulate OpenAI's 1536-dim embeddings

# Create cluster centroids
"""
- For each tag:
  - Get embeddings for each of its 5 anchor words.
  - Compute the **centroid** (mean vector) — this becomes the vector representation of the tag.
"""
cluster_embeddings = {}
for tag, anchor_words in tag_clusters.items():
    anchor_vectors = np.array([dummy_embed(word) for word in anchor_words])
    cluster_embeddings[tag] = np.mean(anchor_vectors, axis=0)

# Embed each word and assign the closest tag(s)
results = []
for word in words:
    word_vector = dummy_embed(word)
    """
    - For each word, compute its **cosine similarity** with every tag's centroid.
    - `cosine_similarity` returns a value between `-1` and `1`, where `1` means very similar.
    """
    similarities = {
        tag: cosine_similarity(word_vector.reshape(1, -1), centroid.reshape(1, -1))[0][0]
        for tag, centroid in cluster_embeddings.items()
    }
    # Select top 1–2 tags based on similarity
    top_tags = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:2]
    results.append((word, [tag for tag, _ in top_tags]))

print(results[:10])  # Show the first 10 results as a preview
