""" Defines the Face repository """

from api.models import Face, db, Embedding
from .Embedding import EmbeddingRepository

class FaceRepository():
    """ The repository for the user model """

    @staticmethod
    def getFaces(user_id):
        """ Query all Faces by user_id """
        return Face.query.filter_by(person=user_id).all()

    @staticmethod
    def getFace(user_id, face_id):
        """ Query a Face by user_id and face_id """
        return Face.query.filter_by(person=user_id, face_id=face_id).first()

    def updateFaceEmbedding(self, embeddings, face_id, user_id, files):
        """ Add embedding to existing face """
        face = self.getFace(user_id, face_id)

        for embedding, file in zip(embeddings, files):
            embedd = EmbeddingRepository.createEmbedding(face_id=face.face_id,
                                                         embedding=embedding,
                                                         file=file)
            face.embeddings.append(embedd)
            embedd.save()

        face.save()

        return face

    def updateFaceDetails(self, user_id, face_id, face_name, face_descr):
        """ Add embedding to existing face """
        face = self.getFace(user_id, face_id)

        if face_name:
            face.face_name = face_name
        if face_descr:
            face.face_descr = face_descr

        face.save()

        return face

    @staticmethod
    def createFace(face_name, face_descr, embeddings, user_id, files):
        """ Create a new face """

        face = Face(
            face_name=face_name,
            face_descr=face_descr
        )

        for embedding, file in zip(embeddings, files):
            embedd = EmbeddingRepository.createEmbedding(face_id=face.face_id,
                                                         embedding=embedding,
                                                         file=file)
            face.embeddings.append(embedd)
        face.person = user_id

        face.save()

        return face
