from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
import os
import shutil
from dotenv import load_dotenv
import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import word_tags


DATA_PATH = "data/"
CHROMA_PATH = "chroma"
load_dotenv()
API_KEY = os.getenv("OPEN_API_KEY")

PROMPT_TEMPLATE = """
From this list of tags: {tags}

Choose the most relevant 1–3 tags for: "{user_input}"

Return just a comma-separated list of the tag names, no brackets, no quotes.
Example: communication, personal, travel
"""




# def main():
#     generate_data_store()

# def generate_data_store():
#     documents = load_documents()
#     chunks = split_text(documents)
#     save_to_chroma(chunks)

# def load_documents():
#     loader = DirectoryLoader(DATA_PATH, glob="*.md")
#     documents = loader.load()
#     return documents


# def split_text(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap = 500,
#         length_function=len,
#         add_start_index=True
#     )

#     chunks = text_splitter.split_documents(documents)
#     # print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

#     document = chunks[10]
#     # print(document.page_content)
#     # print(document.metadata)

#     return chunks


# def save_to_chroma(chunks):
#     # Clear out vector database first.
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)
    
#     # chroma is our vector database that we will be loading the chunks into
#     db = Chroma.from_documents(
#         chunks, OpenAIEmbeddings(api_key=API_KEY), persist_directory=CHROMA_PATH
#     )

#     # db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# hallucinating some tags that are not from the tag list.
def get_tag(word):

    # prepare the DB
    embedding_function = OpenAIEmbeddings(api_key=API_KEY)
    # user_input = word

    # if there are no relevant chunks or snippets of text that match our query, then just return so we dont continue processing.

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(user_input=word, tags=word_tags.tags)

    model = ChatOpenAI(api_key=API_KEY)
    tags = model.predict(prompt)

    tags = [tag.strip() for tag in tags.split(",")]
    
    return tags

# main()
# print(get_tag("expectativas"))

def get_tag_helper(word):
    prompt = f"""
    You are a language learning assistant.

    The user wrote:
    "{word}"

    From this fixed list of tags:
    {', '.join(word_tags.tags)}

    Pick 1–3 tags that best match the user's goal. You MUST choose only from the list above, even if the match is not perfect.
    Do NOT make up new tags. Do NOT return 'none'. Choose the closest options based on meaning or context.
    Return the tags as a comma-separated list (no extra text, no brackets).

    Example output:
    travel, communication
    """

    model = ChatOpenAI(api_key=API_KEY)
    tags = model.predict(prompt)
    tags = [tag.strip() for tag in tags.split(",")]

    return tags


def get_tag_v2(user_input: str) -> list[str]:

    tags = get_tag_helper(user_input)

    tags_set = set(tags)

    print(f"{user_input}   {tags}")
    for tag in tags[:]:
        if tag not in word_tags.tags:
            print(f"{tag} is not in your set of tags ")
            tags_set.remove(tag)
            new_tags = get_tag_helper(tag)
            print(f"new tags: {new_tags}")
            for word in new_tags:
                tags_set.add(word)
  
    
    # filter out anything not in the allowed list
    # return [tag for tag in tags if tag in word_tags.tags]
    return tags_set & set(word_tags.tags)

print(get_tag_v2("inferior"))

"""
blue -> colors -> art
quejas -> complaints -> personal
muriera -> death -> death. 
"""

"""
if a tag is not in the list, call the api on that specific tag to get another tag from that tag.

"""