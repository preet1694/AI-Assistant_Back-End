"""
A robust query service that combines fast,
accurate database lookups with the broad knowledge of the RAG system.
"""
import re
from sqlalchemy.orm import Session, joinedload
from app.db.models import User, Attendance
from app.services.rag_service import get_rag_chain
from typing import Optional

class QueryService:
    """
    Handles user queries by first attempting a fast, structured lookup in the SQL
    database. If that fails or if the query is more general, it falls back to the
    more comprehensive, but slower, RAG service.
    """
    def __init__(self):
        """Initializes the service and pre-loads the RAG chain."""
        print("Initializing QueryService and loading the RAG chain...")
        self.rag_chain = get_rag_chain()
        if self.rag_chain:
            print("RAG chain loaded successfully.")
        else:
            print("CRITICAL: RAG chain could not be loaded. RAG queries will fail.")

    def _find_user_from_query(self, query: str, db: Session) -> Optional[User]:
        """
        Tries to extract a potential student identifier (name, exam no, student id)
        from the user's query and finds the corresponding user in the database.
        """
        # Pattern to find exam numbers (e.g., IT116) or student IDs (e.g., 22ITU...)
        id_pattern = r'\b(IT\d{3}|[A-Z0-9]{10,})\b'
        # A simple pattern to find potential names (capitalized words)
        name_pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'

        potential_id = re.search(id_pattern, query, re.IGNORECASE)
        if potential_id:
            identifier = potential_id.group(1)
            print(f"DEBUG: Found potential ID '{identifier}' in query.")
            # Prioritize exact matches on ID/Exam No
            return db.query(User).filter(
                (User.exam_no.ilike(identifier)) | (User.student_id.ilike(identifier))
            ).first()

        # If no ID is found, look for names.
        potential_names = re.findall(name_pattern, query)
        # Filter out common non-name words
        common_words = {'what', 'is', 'the', 'of', 'for', 'tell', 'me', 'provide'}
        for name in potential_names:
            if name.lower() not in common_words:
                print(f"DEBUG: Found potential name '{name}' in query.")
                # Search for the name in the database
                user = db.query(User).filter(User.name.ilike(f"%{name}%")).first()
                if user:
                    return user # Return the first match
        
        return None

    def handle_query(self, query: str, db: Session, role: str = "guest") -> str:
        """
        Processes a user's query using a hybrid strategy:
        1. Identify if the query is about a specific student.
        2. If so, answer specific questions (like attendance) from the DB.
        3. For other questions about the student (like batch), use the RAG service.
        4. If no student is mentioned, use the RAG service for a general answer.
        """
        query_lower = query.lower()
        user = self._find_user_from_query(query, db)

        # --- Scenario 1: A specific student was identified ---
        if user:
            print(f"DEBUG: Identified user '{user.name}' (Exam No: {user.exam_no}). Analyzing intent...")
            
            # Check for questions that can be answered directly from the database
            if 'attendance' in query_lower:
                if not user.attendance:
                    return f"I couldn't find any attendance records for {user.name}."
                attendance_list = "\n".join([f"- {att.subject}: {att.percentage}%" for att in user.attendance])
                return f"Certainly! Here is the attendance for {user.name}:\n{attendance_list}"

            if 'student id' in query_lower:
                return f"The student ID for {user.name} is {user.student_id}." if user.student_id else f"I don't have a Student ID on file for {user.name}."

            if 'exam no' in query_lower or 'exam number' in query_lower:
                return f"The exam number for {user.name} is {user.exam_no}."

            # If the question is not about DB data, use the RAG service with the student's info.
            # This is where we handle questions like "what is their batch?"
            print(f"DEBUG: Query about '{user.name}' requires RAG lookup. Rephrasing query.")
            
            # Rephrase the query to be more specific for the RAG chain
            rag_query = f"Based on the provided documents, answer this question about the student with exam number {user.exam_no}: {query}"
            
            if not self.rag_chain:
                return "I can access student records, but my broader knowledge base is unavailable right now."
            
            try:
                response = self.rag_chain.invoke(rag_query)
                return response.strip()
            except Exception as e:
                print(f"An error occurred while invoking the RAG chain for a user-specific query: {e}")
                return "I found the student, but had trouble looking up the details. Please try again."

        # --- Scenario 2: No student identified, perform a general RAG search ---
        print("DEBUG: No specific user identified. Using RAG for a general answer.")
        if not self.rag_chain:
            return "I'm sorry, my knowledge base is currently unavailable. Please try again later."
        
        try:
            response = self.rag_chain.invoke(query)
            return response.strip()
        except Exception as e:
            print(f"An error occurred while invoking the RAG chain: {e}")
            return "I encountered an error while processing your request. Please try again."

# --- Integration with your FastAPI Endpoints ---
# This creates a single instance of the service when the application starts.
query_service_instance = QueryService()

def handle_query_logic(query: str, role: str, db: Session):
    """Entry point for the FastAPI endpoint."""
    return query_service_instance.handle_query(query, db, role)
