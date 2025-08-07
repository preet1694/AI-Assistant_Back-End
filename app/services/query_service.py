import re
from sqlalchemy.orm import Session, joinedload
from app.db.models import User, Attendance
from app.services.rag_service import get_rag_response

def find_user_by_id(db: Session, identifier: str):
    """Finds a user by their exam_no or student_id."""
    identifier = identifier.upper()
    return db.query(User).filter(
        (User.exam_no == identifier) | (User.student_id == identifier)
    ).options(joinedload(User.attendance)).first()

def find_user_by_name(db: Session, name: str):
    """
    Finds a user by their full name.
    This performs an exact case-insensitive match for better accuracy.
    """
    return db.query(User).filter(User.name.ilike(name)).first()

def handle_query_logic(query: str, role: str, db: Session):
    """
    Handles user queries by implementing a hybrid search strategy.
    The logic is prioritized to handle natural language first.
    """
    query_lower = query.lower()
    
    # --- Priority 1: Check for relational name queries ---
    if 'after' in query_lower and ('names' in query_lower or 'students' in query_lower):
        print("DEBUG: Relational query keywords ('after', 'names'/'students') detected.")
        try:
            name_to_find = query_lower.split("after", 1)[1].strip()
            print(f"DEBUG: Attempting to find student by name: '{name_to_find}'")
            
            anchor_user = find_user_by_name(db, name_to_find)
            if not anchor_user:
                return f"I'm sorry, I couldn't find a student named '{name_to_find}'. Please check the spelling and provide their full name."

            next_students = db.query(User).filter(User.id > anchor_user.id).order_by(User.id).limit(5).all()
            if not next_students:
                return f"{anchor_user.name} is the last student in the list I have."

            student_list = "\n".join([f"- {s.name} ({s.exam_no})" for s in next_students])
            return f"Of course! Here are the next few students after {anchor_user.name}:\n{student_list}"

        except IndexError:
            print("DEBUG: Relational keywords found, but could not extract a name. Falling back to RAG.")
            return get_rag_response(query)

    # --- Priority 2: Check if the entire query is a student's name ---
    # This runs before the ID check to avoid parts of names being mistaken for IDs.
    print(f"DEBUG: Checking if '{query}' is a student name.")
    user_by_name = find_user_by_name(db, query.strip())
    if user_by_name:
        print(f"DEBUG: Found user '{user_by_name.name}' by full name query.")
        return f"I found a student named {user_by_name.name} (Exam No: {user_by_name.exam_no}). What would you like to know about them? You can ask for their attendance, student ID, etc."

    # --- Priority 3: Check for specific ID lookups ---
    # This regex is now more specific to avoid matching parts of names.
    id_pattern = r'\b(IT\d{3,}|(?=\S*[A-Z])(?=\S*\d)\S{8,})\b'
    potential_ids = re.findall(id_pattern, query, re.IGNORECASE)

    if potential_ids:
        identifier = potential_ids[0]
        print(f"DEBUG: Identifier found: '{identifier}'. Querying SQL database.")
        user = find_user_by_id(db, identifier)
        if user:
            # Handle specific intents for the found user
            if 'name' in query_lower:
                return f"Of course, the name for that ID is {user.name}."
            if 'attendance' in query_lower:
                if not user.attendance:
                    return f"I couldn't find any attendance records for {user.name}."
                attendance_list = "\n".join([f"- {att.subject}: {att.percentage}%" for att in user.attendance])
                return f"Certainly! Here is the attendance for {user.name}:\n{attendance_list}"
            if 'student id' in query_lower:
                return f"The student ID for {user.name} is {user.student_id}." if user.student_id else f"I'm sorry, I don't have a Student ID on file for {user.name}."
            if 'exam no' in query_lower or 'exam number' in query_lower:
                return f"You got it! The exam number for {user.name} is {user.exam_no}."
            return f"That ID belongs to {user.name}. What would you like to know about them?"
        else:
            print(f"DEBUG: Identifier '{identifier}' not in SQL DB. Falling back to RAG.")
            return get_rag_response(query)

    # --- Priority 4: Fallback to RAG for all other general queries ---
    print("DEBUG: No structured query matches found. Using RAG for a general query.")
    return get_rag_response(query)
