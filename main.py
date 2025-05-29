from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models, schemas, crud

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""
so we call this /generate endpoint with the this input: {
  "language": "Spanish",
  "goals": ["travel", "conversation"]
} 

input: schemas.GoalInput validates the input is in the correct format.
the crud/get_vocab_by_goals returns the vocab by returning all the vocab tied to those key words.
    so, "travel" might have some words tied to it,
    "conversation" might have some other words ties to it. 

    this query returns all of those words.


"""

@app.post("/generate", response_model=list[schemas.VocabOut])
def generate_vocab(input: schemas.GoalInput, db: Session = Depends(get_db)):
    vocab_list = crud.get_vocab_by_goals(db, input.language, input.goals)
    print(vocab_list)
    return vocab_list

