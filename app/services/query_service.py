from sqlalchemy.orm import Session
from app.db.models import Attendance
from app.services.rag_service import get_rag_response

def get_student_attendance(db: Session, user_id: int):
    records = db.query(Attendance).filter(Attendance.user_id == user_id).all()
    if not records:
        return "I couldn't find any attendance records for you."
    
    response = "Your attendance is:\n" + "\n".join([f"- {r.subject}: {r.percentage}%" for r in records])
    return response

def handle_query_logic(query: str, role: str, db: Session):
    query_lower = query.lower()
    
    # Intent Detection
    if role == "student" and "attendance" in query_lower:
        user_id = 1 # Hardcoded for now, would come from auth token
        return get_student_attendance(db, user_id)
    else:
        # Fallback to RAG for all other queries
        return get_rag_response(query)