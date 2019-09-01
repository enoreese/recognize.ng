# from api.core import Mixin
from .base import db, BaseModel, MetaBaseModel
from sqlalchemy.types import PickleType, Binary, VARCHAR, VARBINARY
import codecs
from sqlalchemy import func
from sqlalchemy.ext.declarative import declarative_base

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
class Fingerprint(db.Model, DEC_Base, BaseModel, metaclass=MetaBaseModel):
    """Fingerprint Table."""

    __tablename__ = "fingerprint"

    id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    song_hash = db.Column(HashColumn(20))
    offset = db.Column(db.Integer, nullable=False)
    song_id = db.Column(
        db.Integer, db.ForeignKey("song.id", ondelete="SET NULL"), nullable=False
    )

    def __init__(self, song_hash, offset):
        self.song_hash = song_hash
        self.offset = offset

