from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)

    visits = relationship("Visit", back_populates="patient")

class Visit(Base):
    __tablename__ = 'visits'

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    visit_date = Column(String)
    notes = Column(String)

    patient = relationship("Patient", back_populates="visits")

def get_schema_info():
    return {
        "patients": {
            "columns": ["id", "name", "age", "gender"],
            "primary_key": "id"
        },
        "visits": {
            "columns": ["id", "patient_id", "visit_date", "notes"],
            "primary_key": "id",
            "foreign_keys": {
                "patient_id": "patients.id"
            }
        }
    }