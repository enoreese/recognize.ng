from api.core import Mixin
from .base import db
from sqlalchemy.types import PickleType

# Note that we use sqlite for our tests, so you can't use Postgres Arrays
class Embedding(Mixin, db.Model):
    """Embeddings Table."""

    __tablename__ = "embedding"

    id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    embedding = db.Column(PickleType)
    face = db.Column(
        db.Integer, db.ForeignKey("face.id", ondelete="SET NULL"), nullable=True
    )

    def __init__(self, embedding):
        self.embedding = embedding

    def __repr__(self):
        return "<Email {self.embedding}>"
