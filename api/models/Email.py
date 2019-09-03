# from api.core import Mixin
from .base import db, BaseModel, MetaBaseModel

# Note that we use sqlite for our tests, so you can't use Postgres Arrays
class Email(db.Model, BaseModel, metaclass=MetaBaseModel):
    """Email Table."""

    __tablename__ = "email"

    id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    email = db.Column(db.String, nullable=False)
    person = db.Column(
        db.Integer, db.ForeignKey("person.id", ondelete="SET NULL"), nullable=True
    )

    def __init__(self, email):
        self.email = email

    # def __repr__(self):
    #     return "<Email {self.email}>"
