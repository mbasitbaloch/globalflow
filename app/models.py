from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base


class Translation(Base):
    __tablename__ = "translations"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, nullable=True)
    source_text = Column(Text, nullable=False)
    target_text = Column(Text, nullable=False)
    target_lang = Column(String(10), nullable=False)
    content_type = Column(String(50), default="html")
