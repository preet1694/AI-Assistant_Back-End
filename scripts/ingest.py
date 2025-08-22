import os
import re
import sys
import fitz
import camelot
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import List, Dict, Tuple, Optional

# Ensure the app module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.config import settings

# --- Configuration ---
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/"
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'

# --- Helper Functions ---

def load_student_data(file_path: str) -> List[Dict]:
    """Parses the student roll numbers PDF."""
    print(f"Loading student data from: {file_path}")
    students = []
    try:
        doc = fitz.open(file_path)
        text = "".join([page.get_text() for page in doc])
        pattern = re.compile(r'(IT\d{3})[",\s]+?([0-9A-Z]+)\s+([A-Z\s]+)(?=\s*IT\d{3}|$)')
        matches = pattern.findall(text.replace('\n', ' '))
        for exam_no, student_id, name in matches:
            if "ITU" in student_id or "ECU" in student_id:
                students.append({"exam_no": exam_no.strip(), "student_id": student_id.strip(), "name": name.strip().replace("  ", " ")})
            else:
                new_name = f"{student_id} {name}".strip()
                students.append({"exam_no": exam_no.strip(), "student_id": "Not Available", "name": new_name.replace("  ", " ")})
        print(f"Successfully loaded {len(students)} student records.")
        return sorted(students, key=lambda x: int(x['exam_no'].replace("IT", "")))
    except Exception as e:
        print(f"An error occurred while parsing student data: {e}")
        return []

def load_batch_allocations(file_path: str) -> List[Dict]:
    """Parses the batch allocation PDF."""
    print(f"Loading batch allocations from: {file_path}")
    allocations = []
    try:
        doc = fitz.open(file_path)
        text = doc[0].get_text()
        pattern = re.compile(r'([EF]\d)\s+(\d+)\s+(\d+)\s+([A-Z]+)')
        matches = pattern.findall(text)
        for batch, from_id, to_id, counselor in matches:
            allocations.append({"batch": batch, "from_id": int(from_id), "to_id": int(to_id), "counselor": counselor})
        print(f"Loaded {len(allocations)} batch allocation records.")
        return allocations
    except Exception as e:
        print(f"Error loading batch allocations: {e}")
        return []

def get_batch_for_student(exam_no: str, allocations: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """Matches a student's exam number to their allocated batch and counselor."""
    try:
        student_id_num = int(exam_no.replace("IT", ""))
        for alloc in allocations:
            if alloc["from_id"] <= student_id_num <= alloc["to_id"]:
                return alloc["batch"], alloc["counselor"]
    except ValueError:
        return None, None
    return None, None

def parse_timetable_with_camelot(file_path: str, division: str) -> List[Document]:
    """
    Parses a timetable PDF using the Camelot library for robust table extraction.
    This corrected version uses a more flexible logic to handle variations in
    table layouts and headers, and fixes the pandas ambiguity error by using
    index-based access instead of label-based access.
    """
    print(f"Parsing timetable for Division {division} with Camelot from: {file_path}")
    documents = []
    try:
        tables = camelot.read_pdf(file_path, pages='1', flavor='lattice')
        if not tables:
            print(f"Warning: Camelot could not find any tables in {file_path}")
            return []

        df = tables[0].df

        # --- Corrected Header and Data Detection Logic ---

        # First, clean the entire dataframe to handle multi-line cells
        # Use .map() instead of the deprecated .applymap()
        df_cleaned = df.map(lambda x: str(x).replace('\n', ' ').strip())

        data_start_index = -1
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        # Find the first row that contains a day of the week, which marks the start of the actual data
        for i, row in df_cleaned.iterrows():
            # Check the actual string value in the first cell of the row
            if any(day in row.iloc[0] for day in days_of_week):
                data_start_index = i
                break

        if data_start_index == -1:
            print(f"Error: Could not find the starting data row (e.g., 'Monday') for Division {division}.")
            return []

        # The header is the row right before the data starts
        header_row = df_cleaned.iloc[data_start_index - 1]
        data_df = df_cleaned.iloc[data_start_index:]

        # Iterate through the data rows
        for _, row in data_df.iterrows():
            day = row.iloc[0]

            # Special handling for "Saturday" rows which contain multi-line project data
            if "Saturday" in day:
                # Use a regex to find all project entries in the row
                project_pattern = re.compile(r'([EF]\d-PROJECT-II-[A-Z]+)\s+(SW\d+)', re.IGNORECASE)
                full_row_text = ' '.join(row.iloc[1:].astype(str))
                project_matches = project_pattern.findall(full_row_text)

                for project, room in project_matches:
                    content = (
                        f"Timetable Information. Type: Lab. "
                        f"For Division {division} on Saturday, the schedule is a Project. "
                        f"Details: {project} in room {room}."
                    )
                    doc = Document(
                        page_content=content,
                        metadata={"source": os.path.basename(file_path), "record_type": "timetable_entry"}
                    )
                    documents.append(doc)
            else: # Normal weekday processing
                # Skip any rows that are not actual days
                if not any(d in day for d in days_of_week):
                    continue

                # Iterate through the cells of the row, starting from the second cell (index 1)
                for i in range(1, len(row)):
                    details = row.iloc[i]
                    time = header_row.iloc[i] # Get corresponding time from the header row

                    if details and time: # Ensure both details and time slot are present
                        session_type = "Lab" if re.search(r'[EF]\d-', details) else "Lecture"
                        content = (
                            f"Timetable Information. Type: {session_type}. "
                            f"For Division {division} on {day}, during the {time} slot, "
                            f"the schedule is: {details}."
                        )
                        doc = Document(
                            page_content=content,
                            metadata={"source": os.path.basename(file_path), "record_type": "timetable_entry"}
                        )
                        documents.append(doc)

        print(f"Successfully created {len(documents)} timetable documents for Division {division}.")
        return documents
    except Exception as e:
        print(f"An error occurred while parsing timetable {file_path} with Camelot: {e}")
        return []


def chunk_syllabus_pdf(file_path: str) -> List[Document]:
    """Loads and chunks the unstructured syllabus PDF."""
    print(f"Chunking unstructured syllabus from: {file_path}")
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        print(f"Successfully split syllabus into {len(splits)} chunks.")
        return splits
    except Exception as e:
        print(f"Error chunking {file_path}: {e}")
        return []

def create_master_vector_db():
    """
    Orchestrates the entire data ingestion process:
    1. Loads and enriches student data.
    2. Parses timetables using Camelot.
    3. Chunks the syllabus.
    4. Creates and saves a FAISS vector store.
    """
    all_docs = []
    
    # --- 1. Enriched Student Profiles ---
    students = load_student_data(os.path.join(DATA_PATH, "7_Roll Numbers.pdf"))
    allocations = load_batch_allocations(os.path.join(DATA_PATH, "7_IT_2025_BATCH ALLOCATION.pdf"))

    print("\nCreating enriched documents for each student...")
    for student in students:
        exam_no = student["exam_no"]
        batch, counselor = get_batch_for_student(exam_no, allocations)
        content = (f"Student Profile. Full Name: {student['name']}. "
                   f"Exam Number: {exam_no}. "
                   f"Student ID: {student.get('student_id', 'Not Available')}.")
        if batch and counselor:
            content += f" Batch: {batch}. Faculty Counselor: {counselor}."
        doc = Document(page_content=content, metadata={"source": "Multiple Files", "record_type": "student_profile"})
        all_docs.append(doc)
    print(f"Successfully created {len(students)} enriched student profiles.")

    # --- 2. Timetable Parsing (using Camelot) ---
    all_docs.extend(parse_timetable_with_camelot(os.path.join(DATA_PATH, "7_IT_E_2025.pdf"), division="E"))
    all_docs.extend(parse_timetable_with_camelot(os.path.join(DATA_PATH, "7_IT_F_2025.pdf"), division="F"))
    
    # --- 3. Syllabus Chunking ---
    all_docs.extend(chunk_syllabus_pdf(os.path.join(DATA_PATH, "BTech IT 2025-2029 Syllabus File.pdf")))
    
    # --- 4. Vector Store Creation ---
    print(f"\nTotal documents and chunks to be embedded: {len(all_docs)}")
    if not all_docs:
        print("No documents were generated. Aborting vector store creation.")
        return
        
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)
        
    print("Creating and saving master FAISS vector store...")
    db = FAISS.from_documents(all_docs, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"\nMaster vector store created successfully at '{DB_FAISS_PATH}'.")


if __name__ == '__main__':
    # To run this script, execute `python -m scripts.ingest` from the project root.
    create_master_vector_db()