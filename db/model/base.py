from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base

from common.constants import DB_CONNECTION

engine = create_engine(DB_CONNECTION)

Base = declarative_base()
