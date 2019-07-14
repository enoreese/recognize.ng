""" Defines the Face repository """

from api.models import Face, db


class FaceRepository():
    """ The repository for the user model """

    @staticmethod
    def getFaces(user_id):
        """ Query all Faces by user_id """
        return Face.query.filter_by(user_id=user_id).all()

    @staticmethod
    def getFace(user_id, face_id):
        """ Query a Face by user_id and face_id """
        return Face.query.filter_by(user_id=user_id, face_id=face_id).first()

    def updateFaceEmbedding(self, embedding, face_id, user_id):
        """ Add embedding to existing face """
        face = self.getFace(user_id, face_id)

        face.embeddings.append(embedding)

        db.session.add()
        db.session.commit()

        return face

    def updateFaceDetails(self, user_id, face_id, face_name, face_descr):
        """ Add embedding to existing face """
        face = self.getFace(user_id, face_id)

        face.face_name = face_name
        face.face_descr = face_descr

        db.session.add()
        db.session.commit()

        return face

    @staticmethod
    def createFace(face_name, face_descr, embedding, user_id):
        """ Create a new face """
        face = Face(
            face_name=face_name,
            face_descr=face_descr,
            embedding=embedding,
        )

        db.session.add()
        db.session.commit()

        return face
