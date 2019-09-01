# from api.core import Mixin
from .base import db, BaseModel, MetaBaseModel
from sqlalchemy.types import Binary, SmallInteger, VARCHAR, VARBINARY
from sqlalchemy import func
from sqlalchemy.ext.declarative import declarative_base
import codecs

DEC_Base = declarative_base()

class HashColumn(Binary):
    impl = Binary

    def bind_processor(self, dialect):
        """Return a processor that decodes hex values."""

        def process(value):
            return codecs.decode(value, 'hex')

        return process

    def result_processor(self, dialect, coltype):
        """Return a processor that encodes hex values."""

        def process(value):
            return codecs.encode(value, 'hex')

        return process

    def adapt(self, impltype):
        """Produce an adapted form of this type, given an impl class."""
        return HashColumn()


# Note that we use sqlite for our tests, so you can't use Postgres Arrays
class Song(db.Model, DEC_Base, BaseModel, metaclass=MetaBaseModel):
    """Songs Table."""

    __tablename__ = "song"

    id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    song_name = db.Column(db.String(300), nullable=False)
    file_sha1 = db.Column(HashColumn(40))
    fingerprinted = db.Column(SmallInteger, nullable=False, default=0)
    artist = db.Column(
        db.Integer, db.ForeignKey("artist.id", ondelete="SET NULL"), nullable=True
    )
    album = db.Column(db.String(300), nullable=True)

    def __init__(self, song_name, album, file_sha1):
        self.song_name = song_name
        self.album = album
        self.file_sha1 = file_sha1
