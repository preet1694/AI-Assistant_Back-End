import os
import re
import sys
import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.config import settings


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/"
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'

def load_student_data(file_path):
    print(f"Loading student data from: {file_path}")
    students = []
    try:
        doc = fitz.open(file_path)
        text = "".join([page.get_text() for page in doc])
        pattern = re.compile(r'(IT\d{3})[",\s]+?([0-9A-Z]+)\s+([A-Z\s]+)(?=\s*IT\d{3}|$)')
        matches = pattern.findall(text.replace('\n', ' '))
        for exam_no, student_id, name in matches:
            if "ITU" in student_id:
                students.append({"exam_no": exam_no.strip(), "student_id": student_id.strip(), "name": name.strip().replace("  ", " ")})
            else:
                new_name = f"{student_id} {name}".strip()
                students.append({"exam_no": exam_no.strip(), "student_id": "Not Available", "name": new_name.replace("  ", " ")})
        print(f"Successfully loaded {len(students)} student records.")
        return sorted(students, key=lambda x: int(x['exam_no'].replace("IT", "")))
    except Exception as e:
        print(f"An error occurred while parsing student data: {e}")
        return []


def load_batch_allocations(file_path):
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

def get_batch_for_student(exam_no, allocations):
    student_id_num = int(exam_no.replace("IT", ""))
    for alloc in allocations:
        if alloc["from_id"] <= student_id_num <= alloc["to_id"]:
            return alloc["batch"], alloc["counselor"]
    return None, None


def parse_timetable_pdf(file_path, division):
    print(f"Locally parsing timetable for Division {division} from: {file_path}")
    documents = []
    try:
        doc = fitz.open(file_path)
        page = doc[0]
        words = page.get_text("words")

        time_slots = {
            "08:30 AM - 09:30 AM": 100, "09:30 AM - 10:30 AM": 160,
            "10:45 AM - 11:45 AM": 220, "11:45 AM - 12:45 PM": 280,
            "01:30 PM - 02:30 PM": 340, "02:30 PM - 03:30 PM": 400,
            "03:30 PM - 04:30 PM": 460, "04:30 PM - 05:30 PM": 520
        }
        
        lines = {}
        for x0, y0, x1, y1, word, _, _, _ in words:
            y_key = round(y0)
            if y_key not in lines:
                lines[y_key] = []
            lines[y_key].append((x0, word))

        sorted_lines = sorted(lines.items())
        
        current_day = None
        for _, line_words in sorted_lines:
            line_words.sort()
            
            line_text = " ".join(w for _, w in line_words)
            day_match = re.search(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday)\b', line_text)
            if day_match:
                current_day = day_match.group(1)

            if not current_day:
                continue

            for time, x_start in time_slots.items():
                slot_words = [w for x, w in line_words if x_start <= x < x_start + 60]
                if slot_words:
                    details = " ".join(slot_words)
                    session_type = "Lab" if re.search(r'[EF]\d-', details) else "Lecture"
                    content = f"Timetable Information. Type: {session_type}. For Division {division} on {current_day} during {time}, the schedule is: {details}."
                    doc = Document(
                        page_content=content,
                        metadata={"source": os.path.basename(file_path), "record_type": "timetable_entry"}
                    )
                    documents.append(doc)

        print(f"Successfully created {len(documents)} timetable documents for Division {division}.")
        return documents
    except Exception as e:
        print(f"An error occurred while parsing timetable {file_path}: {e}")
        return []


def chunk_syllabus_pdf(file_path):
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
    all_docs = []
    
    students = load_student_data(os.path.join(DATA_PATH, "6_Roll Numbers.pdf"))
    allocations = load_batch_allocations(os.path.join(DATA_PATH, "7_IT_2025_BATCH ALLOCATION.pdf"))

    print("\nCreating enriched documents for each student...")
    for student in students:
        exam_no = student["exam_no"]
        batch, counselor = get_batch_for_student(exam_no, allocations)
        content = (f"Student Profile. Full Name: {student['name']}. Exam Number: {exam_no}. Student ID: {student.get('student_id', 'Not Available')}.")
        if batch and counselor:
            content += f" Batch: {batch}. Faculty Counselor: {counselor}."
        doc = Document(page_content=content, metadata={"source": "Multiple Files", "record_type": "student_profile"})
        all_docs.append(doc)
    print(f"Successfully created {len(students)} enriched student profiles.")

    all_docs.extend(parse_timetable_pdf(os.path.join(DATA_PATH, "7_IT_E_2025.pdf"), division="E"))
    all_docs.extend(parse_timetable_pdf(os.path.join(DATA_PATH, "7_IT_F_2025.pdf"), division="F"))
    all_docs.extend(chunk_syllabus_pdf(os.path.join(DATA_PATH, "BTech IT 2025-2029 Syllabus File.pdf")))
    
    print(f"\nTotal documents and chunks to be embedded: {len(all_docs)}")
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)
    print("Creating and saving master FAISS vector store...")
    db = FAISS.from_documents(all_docs, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"\nMaster vector store created successfully at '{DB_FAISS_PATH}'.")


if __name__ == '__main__':
    create_master_vector_db()