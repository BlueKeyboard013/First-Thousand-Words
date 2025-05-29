from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.dialects.postgresql import ARRAY

from database import Base

"""creates columns in the specified table, if not already created. """

class Vocab(Base):
    __tablename__ = "test"

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, index=True)
    language = Column(String, index=True)
    translation = Column(String)
    example_sentence = Column(String)
    frequency = Column(Float)
    tags = Column(ARRAY(String))  # E.g., ["travel", "business"]