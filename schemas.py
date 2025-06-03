from pydantic import BaseModel
from typing import List

""" used for input and output validations. can create multiple classes to accomodate other API endpoints. """

class GoalInput(BaseModel):
    language: str
    goals: List[str]

class VocabOut(BaseModel):
    word: str
    # translation: str
    # example_sentence: str
    frequency: float
    tags: List[str]

    model_config = {
        "from_attributes": True
    }