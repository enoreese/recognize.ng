# from api.core import Mixin
from .base import db, BaseModel, MetaBaseModel


class Person(db.Model, BaseModel, metaclass=MetaBaseModel):
    """Person Table."""

    __tablename__ = "person"

    id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    password = db.Column(db.String(300))
    phone = db.Column(db.BigInteger, nullable=True)
    fullname = db.Column(db.String, nullable=False)
    emails = db.relationship("Email", backref="emails")

    def __init__(self, fullname, password, phone):
        self.fullname = fullname
        self.password = password
        self.phone = phone


    # def __repr__(self):
    #     return "<Person {self.fullname}>"
