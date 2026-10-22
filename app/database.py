from sqlmodel import Session, SQLModel, create_engine

# This creates a local SQLite file named 'summaries.db' in your project root
DATABASE_URL = "sqlite:///./summaries.db"

# check_same_thread=False is REQUIRED when using SQLite with FastAPI
connect_args = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)


def create_db_and_tables():
    """Creates all SQL tables registered with table=True if they don't exist yet."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """
    Generator that opens a database session for a single request
    and automatically closes it when the request finishes.
    """
    with Session(engine) as session:
        yield session