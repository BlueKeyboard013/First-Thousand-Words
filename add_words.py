from sqlalchemy.orm import Session
from models import Vocab
from database import SessionLocal
from sqlalchemy.dialects.postgresql import ARRAY
from embeddings.get_tags_embeddings import get_tag_v2
from deep_translator import GoogleTranslator



def import_word_frequencies(file_path, language="spanish"):
    db: Session = SessionLocal()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                word, freq = line.strip().split()
                tags_set=list(get_tag_v2(word))  # or ["general"] would be getting this from AI 
                translated_word = GoogleTranslator(source='spanish', target='english').translate(word)
                vocab = Vocab(
                    word=word,
                    language=language,
                    translation=translated_word,  # Placeholder
                    # example_sentence="",  # Placeholder
                    tags = tags_set,
                    frequency=float(freq),
                )
                db.add(vocab)
            except Exception as e:
                print(f"Error with line: {line.strip()} -> {e}")
    db.commit()
    db.close()

# Run it
import_word_frequencies("word_freq.txt")
# def get_tags(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             try:
#                 word, freq = line.strip().split()
#                 get_response(word)
#             except Exception as e:
#                 print(f"Error with line: {line.strip()} -> {e}")
