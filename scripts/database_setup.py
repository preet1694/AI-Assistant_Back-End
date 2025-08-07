"""
Dynamic script to parse a PDF with inconsistent formatting and populate a database.

This script reads the '6_Roll Numbers.pdf' file, intelligently handles
different data formats found across its pages, extracts all relevant student 
identifiers, and populates a SQL database.
"""
import sys
import os
import re
import random

# Add the project root to the Python path to allow for absolute imports
# This allows the script to find the 'app' module.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.database import engine, Base, SessionLocal
from app.db.models import User, Attendance
from langchain_community.document_loaders import PyPDFLoader

# --- Configuration ---
PDF_FILE_PATH = "data/6_Roll Numbers.pdf"

def extract_student_info(raw_string):
    """
    Extracts the student ID and clean name from the raw text.
    e.g., "22ITUON059 AGHERA AAYUSH SURESHBHAI" -> 
          {'student_id': '22ITUON059', 'name': 'AGHERA AAYUSH SURESHBHAI'}
    """
    raw_string = raw_string.strip().strip('"')
    parts = raw_string.split(' ', 1)
    # Check if the first part looks like a valid student ID (alphanumeric)
    if len(parts) == 2 and re.match(r'^[A-Z0-9]+$', parts[0]):
        return {'student_id': parts[0], 'name': parts[1].strip()}
    # Fallback if no student ID is found or format is unusual
    return {'student_id': None, 'name': raw_string.strip()}

def parse_students_from_pdf(file_path):
    """
    Loads a PDF and parses student data using multiple regular expression
    patterns to handle inconsistent formatting.
    """
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at '{file_path}'")
        return []

    print(f"Loading and parsing PDF from '{file_path}'...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    student_records = []
    # Regex for format: "IT001","..."
    pattern1 = re.compile(r'"(IT\d{3})","([^"]+)"')
    # Regex for format: IT126 ... (handles the format on the last page)
    pattern2 = re.compile(r'^(IT\d{3})\s+(.*)$')

    full_text = "".join(doc.page_content for doc in documents)
    
    # Process the text line-by-line to handle different formats
    for line in full_text.split('\n'):
        line = line.strip().replace('\r', '')
        if not line:
            continue

        # Check for both patterns in each line
        match1 = pattern1.search(line)
        match2 = pattern2.match(line)
        
        exam_no, raw_name = None, None
        
        if match1:
            exam_no, raw_name = match1.groups()
        elif match2:
            exam_no, raw_name = match2.groups()

        if exam_no and raw_name:
            # Avoid duplicating records
            if not any(rec['exam_no'] == exam_no for rec in student_records):
                info = extract_student_info(raw_name)
                student_records.append({**info, "exam_no": exam_no})

    if not student_records:
        print("Could not parse any student records. Please check the PDF content and format.")
        return []

    print(f"Successfully parsed {len(student_records)} student records from the PDF.")
    # Sort records to ensure consistent ordering
    return sorted(student_records, key=lambda x: x['exam_no'])

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

        print("Database is empty. Populating with parsed student data...")
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

if __name__ == "__main__":
    # To run this script, execute `python -m scripts.database_setup` from the project root.
    setup_sql_database()
