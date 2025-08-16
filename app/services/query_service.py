"""
A robust query service that combines fast,
accurate database lookups with the broad knowledge of the RAG system.
"""
import re
from sqlalchemy.orm import Session, joinedload
from app.db.models import User, Attendance
from app.services.rag_service import get_rag_chain

class QueryService:
    def __init__(self):
        print("Initializing QueryService and loading the RAG chain...")
        self.rag_chain = get_rag_chain()
        if self.rag_chain:
            print("RAG chain loaded successfully.")
        else:
            print("Error: RAG chain could not be loaded.")

    def _find_user(self, db: Session, identifier: str):
        """Finds a user by their name, exam_no, or student_id."""
        # Search by exam_no or student_id first for exact matches
        user = db.query(User).filter(
            (User.exam_no.ilike(identifier)) | (User.student_id.ilike(identifier))
        ).options(joinedload(User.attendance)).first()
        if user:
            return user
        # If no exact match, search by name (case-insensitive)
        return db.query(User).filter(User.name.ilike(f"%{identifier}%")).options(joinedload(User.attendance)).first()

    def _try_database_lookup(self, query: str, db: Session):
        """
        Attempts to parse the query for specific, fact-based questions
        that can be answered by the database. Returns an answer string if
        successful, otherwise returns None.
        """
        query_lower = query.lower()
        
        # Regex to find potential identifiers (Exam No, Student ID, or a Name)
        # Looks for an ID or a name following keywords like 'of' or 'for'.
        id_pattern = r'\b(IT\d{3}|22ITU\S+)\b'
        name_pattern = r'\b(?:of|for|is|named)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'

        potential_id = re.search(id_pattern, query, re.IGNORECASE)
        potential_name = re.search(name_pattern, query, re.IGNORECASE)

        identifier = None
        if potential_id:
            identifier = potential_id.group(1)
        elif potential_name:
            identifier = potential_name.group(1)
        # As a last resort, treat the whole query as a potential name if it's simple
        elif len(query.split()) <= 3:
             identifier = query
        
        if not identifier:
            return None

        print(f"DEBUG: Identifier '{identifier}' found. Querying SQL database.")
        user = self._find_user(db, identifier)
        
        if user:
            # If a user is found, answer based on the intent in the query
            if 'attendance' in query_lower:
                if not user.attendance:
                    return f"I couldn't find any attendance records for {user.name}."
                attendance_list = "\n".join([f"- {att.subject}: {att.percentage}%" for att in user.attendance])
                return f"Certainly! Here is the attendance for {user.name}:\n{attendance_list}"
            
            if 'student id' in query_lower:
                return f"The student ID for {user.name} is {user.student_id}." if user.student_id else f"I don't have a Student ID on file for {user.name}."

            if 'exam no' in query_lower or 'exam number' in query_lower:
                return f"The exam number for {user.name} is {user.exam_no}."

            # Default response if a user is found but intent is unclear
            return f"I found a student: {user.name} (Exam No: {user.exam_no}). What would you like to know about them? (e.g., 'What is their attendance?')"
        
        return None # No user found in the database

    def handle_query(self, query: str, db: Session, role: str = "guest"):
        """
        Processes a user's query using a hybrid strategy.
        """
        # --- Priority 1: Attempt a direct database lookup ---
        db_answer = self._try_database_lookup(query, db)
        if db_answer:
            print("DEBUG: Answered successfully from the database.")
            return db_answer

        # --- Priority 2: RAG Fallback ---
        print("DEBUG: No structured query match. Using RAG for a general answer.")
        if not self.rag_chain:
            return "I'm sorry, my knowledge base is currently unavailable. Please try again later."
        
        try:
            response = self.rag_chain.invoke(query)
            return response.strip()
        except Exception as e:
            print(f"An error occurred while invoking the RAG chain: {e}")
            return "I encountered an error while processing your request. Please try again."

# --- Integration with your FastAPI Endpoints ---
query_service_instance = QueryService()

def handle_query_logic(query: str, role: str, db: Session):
    """Entry point for the FastAPI endpoint."""
    return query_service_instance.handle_query(query, db, role)