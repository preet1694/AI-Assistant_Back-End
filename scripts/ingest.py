import os
import re
import sys
import fitz
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

def parse_timetable_from_excel(file_path: str, division: str) -> List[Document]:
    """
    Parses a timetable from an Excel file into a list of Documents.
    This method is more robust for structured tabular data.
    """
    print(f"Parsing timetable for Division {division} from Excel: {file_path}")
    documents = []
    try:
        df = pd.read_excel(file_path, header=6)
        
        # --- DEBUGGING: Print the entire DataFrame to see its contents ---
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print("\n--- Raw DataFrame from Excel ---")
        # print(df)
        print("-----------------------------------\n")

        # Clean column names
        df.columns = df.columns.str.strip()
        # print(df.columns);
        # Identify day and time columns
        day_column = "Day" if '' in df.columns else df.columns[1]
        time_columns = [col for col in df.columns if re.match(r'\d{1,2}:\d{2}', str(col))]
        
        print(f"Detected day column: {day_column}")
        print(f"Detected time columns: {time_columns}")

        if not time_columns:
            print("Error: Could not find time columns in the Excel file.")
            return []
            
        for _, row in df.iterrows():
            day = row[day_column]
            if not isinstance(day, str) or not day.strip():
                # --- DEBUGGING: Show which day is being skipped and why ---
                print(f"Skipping row with day: {day}")
                continue
            
            for time_col in time_columns:
                cell_content = row[time_col]
                if pd.notna(cell_content):
                    for line in str(cell_content).split('\n'):
                        line = line.strip()
                        if line:
                            content = (f"Timetable Information. For Division {division} on {day}, "
                                       f"during the {time_col} slot, the schedule is: {line}.")
                            # print(content)
                            documents.append(Document(page_content=content, metadata={"source": os.path.basename(file_path), "record_type": "timetable_entry"}))
        
        print(f"Successfully created {len(documents)} timetable documents for Division {division}.")
        return documents
    except Exception as e:
        print(f"An error occurred while parsing Excel timetable {file_path}: {e}")
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
    2. Parses timetables using the new Excel files.
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

    # --- 2. Timetable Parsing (using the new Excel files) ---
    all_docs.extend(parse_timetable_from_excel(os.path.join(DATA_PATH, "7_E.xlsx"), division="E"))
    all_docs.extend(parse_timetable_from_excel(os.path.join(DATA_PATH, "7_F.xlsx"), division="F"))
    
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