from typing import Dict, Any, List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.models.database import Base, Experiment

class DatabaseManager:
    """Stores every experiment with its exact configuration for reproducibility."""
    def __init__(self, db_url: str = "sqlite:///./signalforge.db"):
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def save_experiment(self, data: Dict[str, Any]):
        session = self.get_session()
        try:
            session.merge(Experiment(**data))
            session.commit()
        finally:
            session.close()

    def get_experiment(self, experiment_id: str):
        session = self.get_session()
        try:
            return session.query(Experiment).filter(Experiment.id == experiment_id).first()
        finally:
            session.close()

    def list_experiments(self, limit: int = 50) -> List[Experiment]:
        session = self.get_session()
        try:
            return session.query(Experiment).order_by(Experiment.created_at.desc()).limit(limit).all()
        finally:
            session.close()
