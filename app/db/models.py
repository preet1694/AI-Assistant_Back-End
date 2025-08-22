# app/db/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    """
    Defines the User model for the database.
    
    This model now includes 'exam_no' and 'student_id' to store all necessary
    identifiers for each student, enabling accurate lookups.
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    role = Column(String, default="student")
    
    # --- ADDED ---
    # Columns to store specific student identifiers from the PDF
    exam_no = Column(String, unique=True, index=True, nullable=False)
    student_id = Column(String, unique=True, index=True, nullable=True)
    
    # Relationship to the Attendance model
    attendance = relationship("Attendance", back_populates="user")

class Attendance(Base):
    """Defines the Attendance model for the database."""
    __tablename__ = 'attendance'

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String, index=True)
    percentage = Column(Float)
    user_id = Column(Integer, ForeignKey('users.id'))

    # --- ADDED ---
    # Defines the other side of the relationship
    user = relationship("User", back_populates="attendance")
