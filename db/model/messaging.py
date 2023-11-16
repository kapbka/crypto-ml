from sqlalchemy import Column, Sequence
from sqlalchemy import Integer, LargeBinary, DateTime, String, Boolean
from sqlalchemy.sql import func

from db.model.base import Base


class Payload(Base):
    __tablename__ = 'payload'

    id = Column(Integer, Sequence('payload_id_seq'), primary_key=True)
    data = Column(LargeBinary, nullable=False)
    producer = Column(String, nullable=False)
    consumer = Column(String, nullable=False)
    available = Column(Boolean, nullable=False)

    create_ts = Column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = ({'extend_existing': True})


class Consumer(Base):
    __tablename__ = 'consumer'

    id = Column(String, primary_key=True)
    last_processed_payload = Column(Integer)
    active = Column(Integer, nullable=False)

    create_ts = Column(DateTime, server_default=func.now(), nullable=False)
    update_ts = Column(DateTime, onupdate=func.now())

    __table_args__ = ({'extend_existing': True})

    def __repr__(self):
        return f"<Consumer({self.id}), last: {self.last_processed_payload}>"
