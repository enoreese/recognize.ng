""" Defines the Embedding repository """

from api.models import Embedding


class EmbeddingRepository():
    """ The repository for the user model """

    @staticmethod
    def getEmbeddings(face_id):
        """ Query all Embeddings by face_id """
        return Embedding.query.filter_by(face_id=face_id).all()

    @staticmethod
    def getEmbedding(face_id, embedding_id):
        """ Query a Embedding by user_id and face_id """
        return Embedding.query.filter_by(embedding_id=embedding_id, face_id=face_id).first()

    @staticmethod
    def createEmbedding(face_id, embedding):
        """ Create a new face embedding """
        face = Embedding(
            embedding=embedding,
        )

        db.session.add()
        db.session.commit()

        return face
