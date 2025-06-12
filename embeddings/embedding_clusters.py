import openai
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPEN_API_KEY")

# Your word list
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

# Step 1: Get embeddings
response = openai.embeddings.create(
    input=words,
    model="text-embedding-3-small"
)

embeddings = [r.embedding for r in response.data]

# Step 2: Cluster with KMeans
num_clusters = 8  # You can tweak this
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Step 3: Group words by cluster
clustered_words = defaultdict(list)
for word, label in zip(words, clusters):
    clustered_words[label].append(word)

# Step 4: Generate labels for each cluster using new OpenAI chat completion API
for cluster_id, word_list in clustered_words.items():
    print(f"\nCluster {cluster_id} ({len(word_list)} words):")
    print(", ".join(word_list))
    
    label_prompt = f"Give a concise 1-2 word label summarizing this group of words: {', '.join(word_list)}"
    
    label_resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes word groups."},
            {"role": "user", "content": label_prompt}
        ]
    )
    
    label = label_resp.choices[0].message.content.strip()
    print("Label:", label)