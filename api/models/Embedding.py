# from api.core import Mixin
from .base import db, BaseModel, MetaBaseModel
from sqlalchemy.types import PickleType

# Note that we use sqlite for our tests, so you can't use Postgres Arrays
class Embedding(db.Model, BaseModel, metaclass=MetaBaseModel):
    """Embeddings Table."""

    __tablename__ = "embedding"

    id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    embedding = db.Column(PickleType)
    file = db.Column(
        db.Integer, db.ForeignKey("files.id", ondelete="SET NULL"), nullable=True
    )
    face = db.Column(
        db.Integer, db.ForeignKey("face.face_id", ondelete="SET NULL"), nullable=True
    )

    def __init__(self, embedding):
        self.embedding = embedding
