"""
One-time script to set up the database.

This script creates the necessary tables in the database based on the defined models
and populates it with initial mock data if the database is empty.
"""
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.database import engine, Base, SessionLocal
from app.db.models import User, Attendance

def setup_database():
    """Creates tables and populates them with initial data."""
    # Create all tables defined in the models
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if data already exists to prevent duplicates
        if db.query(User).count() == 0:
            print("Database is empty. Populating with mock data...")
            # Add a mock student user
            student_user = User(id=1, name="Alex", role="student")
            db.add(student_user)
            db.commit()

            # Add attendance records for the mock student
            db.add(Attendance(subject="Physics", percentage=85.0, user_id=1))
            db.add(Attendance(subject="Chemistry", percentage=92.0, user_id=1))
            db.commit()
            print("Database populated successfully.")
        else:
            print("Database already contains data. Skipping population.")
    finally:
        db.close()

if __name__ == "__main__":
    setup_database()