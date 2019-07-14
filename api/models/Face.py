"""
Define the Face model
"""
from api.core import Mixin
from .base import db
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import PickleType


class Face(Mixin, db.Model):
    """ The Face model """

    __tablename__ = "face"

    face_name = db.Column(db.String(300))
    face_descr = db.Column(db.String(300))
    person = db.Column(
        db.Integer, db.ForeignKey("person.id", ondelete="SET NULL"), nullable=True
    )
    face_id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    embeddings = db.relationship("Embedding", backref="embeddings")

    def __init__(self, face_name, face_descr, embedding=None):
        """ Create a new face """
        self.face_name = face_name
        self.face_descr = face_descr
        self.embedding = embedding
