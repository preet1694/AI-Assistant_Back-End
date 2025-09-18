"""
A robust query service that combines fast,
accurate database lookups with the broad knowledge of the RAG system.
"""
import re
from sqlalchemy.orm import Session, joinedload
from app.db.models import User, Attendance
from app.services.rag_service import get_rag_chain
from typing import Optional
from datetime import datetime, time

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
    
    def _find_timetable_entry(self, query: str) -> Optional[str]:
        """
        Extracts day and time from the query and performs a RAG search
        to find the corresponding timetable entry.
        """
        time_pattern = r'(?i)(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:at)?\s*(\d{1,2}(?::\d{2})?)\s*(?:am|pm)?'
        match = re.search(time_pattern, query)

        if not match:
            return None # No time/day pattern found in the query

        query_day = match.group(1).capitalize()
        query_time_str = match.group(2)
        
        print(f"DEBUG: Detected day: {query_day}, Detected time: {query_time_str}")

        try:
            if 'am' in query.lower() or 'pm' in query.lower():
                query_time_obj = datetime.strptime(query_time_str + re.search(r'(am|pm)', query, re.I).group(1), '%I:%M%p').time()
            else:
                query_time_obj = datetime.strptime(query_time_str, '%H:%M').time()
        except ValueError:
            return "I couldn't understand the time format. Please try '9:00 AM' or '09:00'."

        # Create a semantic query for the retriever
        rag_query = f"Timetable for {query_day} at {query_time_str}"
        
        # --- FIX: Access the retriever correctly from your RAG chain ---
        # The retriever should be a part of your RAG chain setup in rag_service.py
        # If your rag_chain is a simple sequence, you may need to load the vector store and retriever here.
        
        try:
            from app.services.rag_service import get_retriever
            relevant_docs = self.rag_chain.invoke(rag_query)
        except ImportError:
            # Fallback if get_retriever is not available
            return "I am unable to access my timetable knowledge base at the moment."
        except Exception as e:
            print(f"An error occurred during retriever search: {e}")
            return "I encountered an error searching the timetable. Please try again."

        for doc in relevant_docs:
            doc_match = re.search(r'For\s(.*?),\s+during\sthe\s(.*?)\sTO\s(.*?)\sslot', doc)
            if doc_match and doc_match.group(1).strip() == query_day:
                doc_start_time_str, doc_finish_time_str = doc_match.group(2), doc_match.group(3)
                
                try:
                    doc_start_time_obj = datetime.strptime(doc_start_time_str, '%I:%M').time()
                    doc_finish_time_obj = datetime.strptime(doc_finish_time_str, '%I:%M').time()
                    
                    if doc_start_time_obj <= query_time_obj < doc_finish_time_obj:
                        return doc
                except ValueError:
                    continue

        return f"No timetable information found for {query_day} at {query_time_str}."
    
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

        # Check for timetable-specific queries first
        timetable_response = self._find_timetable_entry(query)
        if timetable_response:
            return timetable_response
        
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
            print(f"DEBUG: Query about '{user.name}' requires RAG lookup. Rephrasing query.")
            
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
query_service_instance = QueryService()

def handle_query_logic(query: str, role: str, db: Session):
    """Entry point for the FastAPI endpoint."""
    return query_service_instance.handle_query(query, db, role)