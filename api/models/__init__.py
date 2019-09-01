# this file structure follows http://flask.pocoo.org/docs/1.0/patterns/appfactories/
# initializing db in api.models.base instead of in api.__init__.py
# to prevent circular dependencies
from .Email import Email
from .Person import Person
from .base import db
from .Embedding import Embedding
from .Face import Face
from .Fingerprint import Fingerprint
from .Song import Song
from .Artist import Artist
from .Files import Files

__all__ = ["Email",
           "Person",
           "db",
           "Embedding",
           "Face",
           "Fingerprint",
           "Song",
           "Artist",
           "Files"]

# You must import all of the new Models you create to this page
