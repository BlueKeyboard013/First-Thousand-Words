from sqlalchemy.orm import Session
from models import Vocab
from database import SessionLocal
from sqlalchemy.dialects.postgresql import ARRAY
from get_tags import get_tag

def import_word_frequencies(file_path, language="spanish"):
    db: Session = SessionLocal()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                word, freq = line.strip().split()
                vocab = Vocab(
                    word=word,
                    language=language,
                    # translation="",  # Placeholder
                    # example_sentence="",  # Placeholder
                    frequency=float(freq),
                    tags=list(get_tag(word))  # or ["general"] would be getting this from AI 
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
