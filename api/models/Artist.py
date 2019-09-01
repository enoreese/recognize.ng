# from api.core import Mixin
from .base import db, BaseModel, MetaBaseModel

# Note that we use sqlite for our tests, so you can't use Postgres Arrays
class Artist(db.Model, BaseModel, metaclass=MetaBaseModel):
    """Artist Table."""

    __tablename__ = "artist"

    id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    artist_name = db.Column(db.String(50), nullable=False)
    artist_description = db.Column(db.String(300), nullable=False)
    songs = db.relationship("Song", backref="songs")
    record_label = db.Column(db.String(300), nullable=True)


    def __init__(self, artist_name, artist_description, record_label):
        self.artist_name = artist_name
        self.artist_description = artist_description
        self.record_label = record_label