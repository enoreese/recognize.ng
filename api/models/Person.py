from api.core import Mixin
from .base import db


class Person(Mixin, db.Model):
    """Person Table."""

    __tablename__ = "person"

    id = db.Column(db.Integer, unique=True, primary_key=True, autoincrement=True)
    password = db.Column(db.String(300))
    phone = db.Column(db.BigInteger, nullable=True)
    fullname = db.Column(db.String, nullable=False)
    emails = db.relationship("Email", backref="emails")
    faces = db.relationship("Face", backref="faces")

    def __init__(self, fullname, pasword, phone):
        self.fullname = fullname
        self.password = pasword
        self.phone = phone


    def __repr__(self):
        return "<Person {self.fullname}>"
