from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.query_service import handle_query_logic

app = FastAPI(title="College AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    role: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the College AI Assistant API"}

@app.post("/api/query")
def handle_query(request: QueryRequest, db: Session = Depends(get_db)):
    response_text = handle_query_logic(request.query, request.role, db)
    return {"answer": response_text}