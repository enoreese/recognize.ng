from flask import Blueprint, request
from api.models import db, Person, Email
from api.repositiory import PersonRepository, EmbeddingRepository, FaceRepository
from api.core import create_response, serialize_list, logger
from sqlalchemy import inspect
from api.utils import img_to_encoding, create_model, ml_func

MODEL = create_model()

main = Blueprint("face", __name__)  # initialize blueprint


# function that is called when you visit /
@main.route("/face")
def index():
    # you are now in the current application context with the main.route decorator
    # access the logger with the logger from api.core and uses the standard logging module
    # try using ipdb here :) you can inject yourself
    logger.info("Hello World!")
    return "<h1>Hello World!</h1>"


# function that is called when you visit /persons
@main.route("/face/<user_id>/<face_id>", methods=["GET"])
def get_face():
    user_id = request.view_args['user_id']
    face_id = request.view_args['face_id']
    face = FaceRepository.getFace(user_id=user_id, face_id=face_id)
    embedding = EmbeddingRepository.getEmbedding(face_id=face_id, embedding_id=face.embedding.id)

    response = {
        'status': 'success',
        'data': {
            'face_name': face.face_name,
            'face_descr': face.face_descr,
            'embedding': embedding.embedding
        }
    }

    return create_response(data={"response": serialize_list(response)})

# function that is called when you visit /persons
@main.route("/face/<user_id>/", methods=["GET"])
def get_faces():
    user_id = request.view_args['user_id']
    faces = FaceRepository.getFaces(user_id=user_id)
    embeddings = []

    for face in faces:
        embedding = EmbeddingRepository.getEmbedding(face_id=face.face_id, embedding_id=face.embedding.id)
        embeddings.append({
            'face_name': face.face_name,
            'face_descr': face.face_descr,
            'embedding': embedding.embedding
        })

    response = {
        'status': 'success',
        'data': embeddings
    }

    return create_response(data={"response": serialize_list(response)})


# POST request for /face
@main.route("/face", methods=["POST"])
def create_face():
    data = request.get_json()

    logger.info("Data recieved: %s", data)
    if "person_id" not in data:
        msg = "No name provided for face."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "face_name" not in data:
        msg = "No name provided for face."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "face_descr" not in data:
        msg = "No description provided for face."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "images" not in data:
        msg = "No password provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if len(data['images']) > 3:
        msg = "Only 3 images is allowed per face."
        logger.info(msg)
        return create_response(status=422, message=msg)


    person = PersonRepository.getById(id=data['user_id'])
    # create SQLAlchemy Objects
    new_face = FaceRepository.createFace(face_name=data['face_name'], face_descr=data['face_descr'])
    for image in data['images']:
        embed = img_to_encoding(image, MODEL)
        embedding = EmbeddingRepository.createEmbedding(face_id=new_face.face_id, embedding=embed)
        new_face.embeddings.append(embedding)

        person = PersonRepository.getById(id=data['user_id'])
        person.faces.append(new_face)

    # commit it to database
    # db.session.add_all([new_face, embedding, person])
    # db.session.commit()
    return create_response(
        status='success',
        message= "Successfully created face {new_face.name} with id: {new_face.face_id}",
        data={
            'person': {
                'fullname': person.fullname,
                'id': person.id
            },
            'face': {
                'face_id': new_face.face_id,
                'face_name': new_face.face_name,
                'face_descr': new_face.face_descr,
                'embeddings': embedding.embedding
            }
        }
    )


# PUT request for /face
@main.route("/face", methods=["PUT"])
def update_face():
    data = request.get_json()

    updated_face = None
    embedding = []

    logger.info("Data recieved: %s", data)
    if "face_id" not in data:
        msg = "No id provided for face to update."
        logger.info(msg)
        return create_response(status=422, message=msg)

    if "face_id" not in data:
        msg = "No id provided for face to update."
        logger.info(msg)
        return create_response(status=422, message=msg)

    if "face_descr" in data:
        # msg = "No email provided for person."
        # logger.info(msg)
        # return create_response(status=422, message=msg)
        updated_face = FaceRepository.updateFaceDetails(face_id=data['face_id'],face_descr=data['face_descr'])

    if "images" in data:
        # msg = "No password provided for person."
        # logger.info(msg)
        # return create_response(status=422, message=msg)
        len_images = len(data['images'])
        if len_images > 0:
            if len_images > 3:
                msg = "Only 3 images is allowed per face."
                logger.info(msg)
                return create_response(status=422, message=msg)

            face = FaceRepository.getFace(user_id=data['user_id'], face_id=data['face_id'])
            emebddings = face.embeddings[len_images:]

            for image in data['images']:
                embedding = img_to_encoding(image, model=MODEL)
                updated_face = FaceRepository.updateFaceEmbedding(face_id=data['face_id'], embedding=embedding)


    # commit it to database
    db.session.add_all([updated_face, email])
    db.session.commit()
    return create_response(
        status='success',
        message= "Successfully created face {new_face.name} with id: {new_face.face_id}",
        data={
            'face': {
                'face_id': updated_face.face_id,
                'face_name': updated_face.face_name,
                'face_descr': updated_face.face_descr,
                'embeddings': embedding
            }
        }
    )

# POST request for /face/recognize
@main.route("/face/recognize", methods=["POST"])
def recognize_face():
    data = request.get_json()

    logger.info("Data recieved: %s", data)
    if "user_id" not in data:
        msg = "No user id provided for face recognition."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "image" not in data:
        msg = "No image provided for face to update."
        logger.info(msg)
        return create_response(status=422, message=msg)

    faces = FaceRepository.getFaces(user_id=data['user_id'])

    database = {}

    for face in faces:
        embedding = EmbeddingRepository.getEmbedding(face_id=face.face_id, embedding_id=face.embedding.id)
        database[face.face_id] = embedding

    min_dist, identity, message = ml_func.who_is_it(image_path=data['image'], model=MODEL)

    if message == "No face found":
        logger.info("Cannot recognize face ", identity, min_dist)
        return create_response(
            status='success',
            message=message,
            data={}
        )
    else:
        logger.info("Recognized identity id: %s with distance of: ", identity, min_dist)
        face = FaceRepository.getFace(data['user_id'], face_id=identity)

        return create_response(
            status='success',
            message=message,
            data={
                "face": {
                    "face_id": face.face_id,
                    "face_name": face.face_name,
                    "face_descr": face.face_descr,
                    "distance": min_dist
                }
            }
        )

    # POST request for /face/recognize
@main.route("/face/verify", methods=["POST"])
def verify_face():
    data = request.get_json()

    logger.info("Data recieved: %s", data)
    if "face_id" not in data:
        msg = "No face id provided for face you'd like to verify."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "user_id" not in data:
        msg = "No user id provided."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "image" not in data:
        msg = "No image provided for face to verify."
        logger.info(msg)
        return create_response(status=422, message=msg)

    face = FaceRepository.getFace(face_id=data['face_id'], user_id=data['user_id'])

    database = {}

    for embedd in face.embeddings:
        embedding = EmbeddingRepository.getEmbedding(face_id=face.face_id, embedding_id=embedd.id)
        database[embedd.id] = embedding

    dist, match = ml_func.verify(data['image'], database, MODEL)

    if not match:
        logger.info("Un-Verified identity with distance of: ", dist)
        return create_response(
            status='success',
            message='Image does not match face',
            data={}
        )
    else:
        logger.info("Verified identity with distance of: ", dist)

        face = FaceRepository.getFace(data['user_id'], face_id=identity)

        return create_response(
            status='success',
            message='Image matches the face',
            data={
                "face": {
                    "face_id": face.face_id,
                    "face_name": face.face_name,
                    "face_descr": face.face_descr,
                    "diatance": dist
                }
            }
        )

        # POST request for /face/recognize

@main.route("/face/detect", methods=["POST"])
def verify_face():
    data = request.get_json()

    if "image" not in data:
        msg = "No image provided to detect face."
        logger.info(msg)
        return create_response(status=422, message=msg)

    detected, faces, message, eyes = ml_func.detect_face(data['image'])

    if not detected:
        logger.info(message)
        return create_response(
            status='success',
            message=message,
            data={}
        )
    else:
        logger.info(message)

        return create_response(
            status='success',
            message=message,
            data={
                "face": {
                    "face_coordinates": faces,
                    "eyes_coordinates": eyes
                }
            }
        )

    # commit it to database
    # db.session.add_all([new_person, email])
    # db.session.commit()


