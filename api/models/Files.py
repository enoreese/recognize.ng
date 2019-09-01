# from api.core import Mixin
from .base import db, BaseModel, MetaBaseModel
from sqlalchemy.types import PickleType

# Note that we use sqlite for our tests, so you can't use Postgres Arrays
class Files(db.Model, BaseModel, metaclass=MetaBaseModel):
    """Files Table."""

    __tablename__ = "files"

    id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    file_url = db.Column(db.String(300))
    storage = db.Column(db.String(300))

    def __init__(self, storage, file_url):
        self.storage = storage
        self.file_url = file_url
