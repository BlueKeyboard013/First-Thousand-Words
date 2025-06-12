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
    # print(f"embedding {text}")
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return [r.embedding for r in response.data]

# embeddings = [get_embedding(word) for word in words]

embeddings = get_embedding(words)

# ➤ Convert to NumPy array before t-SNE
embeddings_2d = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(np.array(embeddings))

# Plot
plt.figure(figsize=(10, 6))
for i, word in enumerate(words):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.5, y + 0.5, word, fontsize=9)
plt.title("2D Visualization of Word Embeddings (t-SNE)")
plt.grid(True)
plt.show()