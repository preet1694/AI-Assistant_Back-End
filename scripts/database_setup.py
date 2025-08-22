"""
Dynamic script to parse a PDF with inconsistent formatting and populate a database.

This script reads the '7_Roll Numbers.pdf' file, intelligently handles
different data formats found across its pages, extracts all relevant student 
identifiers, and populates a SQL database.
"""
import sys
import os
import re
import random
import fitz

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.database import engine, Base, SessionLocal
from app.db.models import User, Attendance

PDF_FILE_PATH = "data/7_Roll Numbers.pdf"

def extract_student_info(raw_string):
    """
    Extracts the student ID and clean name from the raw text.
    (This function is unchanged as its logic is sound.)
    """
    raw_string = raw_string.strip().strip('"')
    parts = raw_string.split(' ', 1)
    if len(parts) == 2 and re.match(r'^[A-Z0-9]+$', parts[0]):
        return {'student_id': parts[0], 'name': parts[1].strip()}
    return {'student_id': "Not Available", 'name': raw_string.strip()}

def parse_students_from_pdf(file_path: str):
    """
    Loads a PDF and parses student data using robust, page-by-page preprocessing
    to handle repeating headers and ensure all records are extracted.
    """
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at '{file_path}'")
        return []

    print(f"Loading and parsing PDF from '{file_path}'...")
    students = []
    try:
        doc = fitz.open(file_path)
        
        # 1. PREPROCESSING STEP: Clean the text by removing page headers
        cleaned_text = ""
        for page in doc:
            page_text = page.get_text()
            lines = page_text.split('\n')
            # Filter out the header line from each page
            cleaned_lines = [line for line in lines if "Exam No." not in line]
            cleaned_text += "\n".join(cleaned_lines)

        # 2. PARSING STEP: Use a robust regex on the cleaned text
        pattern = re.compile(r'(IT\d{3})[",\s]+?([0-9A-Z]+)\s+([A-Z\s]+)(?=\s*IT\d{3}|$)')
        matches = pattern.findall(cleaned_text.replace('\n', ' '))
        
        for exam_no, student_id, name in matches:
            # Handle cases where the student ID is part of the name field
            if "ITU" in student_id or "ECU" in student_id:
                students.append({
                    "exam_no": exam_no.strip(), 
                    "student_id": student_id.strip(), 
                    "name": name.strip().replace("  ", " ")
                })
            else:
                # If the second part isn't a student ID, it's part of the name
                new_name = f"{student_id} {name}".strip()
                students.append({
                    "exam_no": exam_no.strip(), 
                    "student_id": "Not Available", 
                    "name": new_name.replace("  ", " ")
                })

        if not students:
            print("Could not parse any student records.")
            return []

        print(f"Successfully parsed {len(students)} student records from the PDF.")
        return sorted(students, key=lambda x: int(x['exam_no'].replace("IT", "")))

    except Exception as e:
        print(f"An error occurred while parsing student data: {e}")
        return []


def setup_sql_database():
    """
    Orchestrates the database setup: creates tables and populates them with
    all necessary student identifiers for accurate lookups.
    """
    print("Setting up SQL database...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        if db.query(User).count() > 0:
            print("Database already contains data. Skipping population.")
            return

        student_data = parse_students_from_pdf(PDF_FILE_PATH)
        if not student_data:
            print("Aborting database setup due to parsing failure.")
            return

        print(f"Database is empty. Populating with parsed student data...")
        for student in student_data:
            user_id = int(re.search(r'\d+', student["exam_no"]).group())
            
            student_user = User(
                id=user_id,
                name=student["name"],
                exam_no=student["exam_no"],
                student_id=student["student_id"],
                role="student"
            )
            db.add(student_user)
            db.flush()

            subjects = ["Physics", "Chemistry", "Mathematics", "Data Structures", "Algorithms"]
            for subject in subjects:
                db.add(Attendance(
                    subject=subject,
                    percentage=round(random.uniform(65.0, 99.5), 2),
                    user_id=student_user.id
                ))

        db.commit()
        print(f"Database populated successfully with {len(student_data)} students.")

    except Exception as e:
        print(f"An error occurred during database setup: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == '__main__':
    # To run this script, execute python -m scripts.database_setup from the project root.
    setup_sql_database()