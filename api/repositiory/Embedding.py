""" Defines the Embedding repository """

from api.models import Embedding, db


class EmbeddingRepository():
    """ The repository for the user model """

    @staticmethod
    def getEmbeddings(face_id):
        """ Query all Embeddings by face_id """
        return Embedding.query.filter_by(face=face_id).all()

    @staticmethod
    def getEmbedding(face_id, embedding_id):
        """ Query a Embedding by user_id and face_id """
        return Embedding.query.filter_by(id=embedding_id, face=face_id).first()

    @staticmethod
    def createEmbedding(face_id, embedding, file):
        """ Create a new face embedding """
        embedding = Embedding(
            embedding=embedding,
        )
        embedding.face = face_id
        embedding.file = file

        embedding.save()

        return embedding
