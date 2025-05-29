from sqlalchemy.orm import Session
from models import Vocab

"""queries the vocab words based on the tags and input goals.   """
"""“Give me up to limit vocabulary words in the specified language that match at least one of the goals tags, sorted by frequency (most common first).”"""

def get_vocab_by_goals(db: Session, language: str, goals: list, limit=500):
    return db.query(Vocab)\
        .filter(Vocab.language == language)\
        .filter(Vocab.tags.overlap(goals))\
        .order_by(Vocab.frequency.desc())\
        .limit(limit)\
        .all()