import numpy as np
import asyncio
from flask import Blueprint, request
from api.repositiory import PersonRepository, EmbeddingRepository, FaceRepository, FileRepository
from api.core import create_response, serialize_list, logger, serialize_embeddings
from api.utils import img_to_encoding, train_knn, predict_face, handle_upload, create_model, who_is_it, verify, detect_face, who_is_it_bulk
from api import MODEL

STORAGE = 'cloudinary'

loop = asyncio.get_event_loop()

main = Blueprint("face", __name__)  # initialize blueprint

# function that is called when you visit /persons
@main.route("/face", methods=["GET"])
def get_face():
    user_id = request.args.get('user_id')
    face_id = request.args.get('face_id')
    face = FaceRepository.getFace(user_id=user_id, face_id=face_id)
    embeddings = []
    for embed in face.embeddings:
        embedding = EmbeddingRepository.getEmbedding(face_id=face_id, embedding_id=embed.id)
        embeddings.append(embedding.embedding)

    logger.info("Face Object: %s", face.json)
    face = face.json

    response = {
        'status': 'success',
        'data': {
            'face_name': face['face_name'],
            'face_descr': face['face_descr'],
            'embedding': serialize_embeddings(embeddings)
        }
    }

    return create_response(status=200, data={"response": response})

# function that is called when you visit /persons
@main.route("/faces", methods=["GET"])
def get_faces():
    user_id = request.args.get('user_id')
    faces = FaceRepository.getFaces(user_id=user_id)
    face_obj = []

    logger.info("Faces Object: %s", faces)

    for face in faces:
        obj = {}
        obj['face_name'] = face.face_name
        obj['face_id'] = face.face_id
        obj['face_descr'] = face.face_descr
        embeddings = []
        for embedd in face.embeddings:
            embedding = EmbeddingRepository.getEmbedding(face_id=face.face_id, embedding_id=embedd.id)
            embeddings.append(embedding.embedding)
        obj['embeddings'] = serialize_embeddings(embeddings)
        face_obj.append(obj)

    return create_response(status=200, data={"response": serialize_list(face_obj)})


# POST request for /face
@main.route("/face", methods=["POST"])
def create_face():
    data = dict(request.form)
    images = request.files.getlist("images")

    logger.info("Data recieved: %s", data)
    if "user_id" not in data:
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
    if not images:
        msg = "No images provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)

    person = PersonRepository.getById(id=data['user_id'])
    logger.info("Person Object: %s", person.json)
    # keys = np.empty((0,1), float)
    # values = np.empty((0,128), float)
    embeddings = []
    uploaded_images = []

    for image in images:
        # save file to storage
        prefix = str(data['user_id']) + '-create-face'
        uploaded_image = handle_upload(image, data=data, bucket='faces', prefix=prefix, storage=STORAGE)
        # save file to db
        file = FileRepository.createFile(file_url=uploaded_image, storage=STORAGE)
        uploaded_images.append(file.id)
        # generate embeddings
        status, embed = img_to_encoding(uploaded_image, MODEL, resize=True)
        if not status:
            logger.info(embed)
            return create_response(status=422, message=embed)
        embeddings.append(embed)
        # np.append(keys, int(data['user_id']), axis=0)
        # np.append(values, embed, axis=0)
        # keys.append(int(data['user_id']))

        # values.append(embed)

    # print("keys: ", keys)
    # print("values: ", values)

    new_face = FaceRepository.createFace(face_name=data['face_name'],
                                         face_descr=data['face_descr'],
                                         embeddings=embeddings,
                                         user_id=data['user_id'],
                                         files=uploaded_images)
    # TODO: Train new KNN on new images
    loop.run_until_complete(train_knn(user_id=data['user_id']))
    # await train_knn(keys=keys, values=values, user_id=data['user_id'])

    logger.info("Face Object: %s", new_face.json)
    new_face = new_face.json
    person = person.json

    return create_response(
        status=200,
        message="Successfully created face {new_face.face_name} with id: {new_face.face_id}",
        data={
            'person': {
                'fullname': person['fullname'],
                'id': person['id']
            },
            'face': {
                'face_id': new_face['face_id'],
                'face_name': new_face['face_name'],
                'face_descr': new_face['face_descr'],
                'embeddings': serialize_embeddings(embeddings)
            }
        }
    )


# PUT request for /face
@main.route("/face", methods=["PUT"])
def update_face():
    data = dict(request.form)
    images = request.files.getlist("images")

    updated_face = None
    embeddings = []
    keys = []
    values = []

    logger.info("Data recieved: %s", data)
    if "user_id" not in data:
        msg = "No user id provided."
        logger.info(msg)
        return create_response(status=422, message=msg)

    if "face_id" not in data:
        msg = "No id provided for face to update."
        logger.info(msg)
        return create_response(status=422, message=msg)

    if "face_descr" in data or "face_name" in data:
        face_repo = FaceRepository()
        updated_face = face_repo.updateFaceDetails(
            face_name=data['face_name'],
            face_id=data['face_id'],
            face_descr=data['face_descr'],
            user_id=data['user_id'])

        # TODO: optimize this loop
        for embed in updated_face.embeddings:
            embedding = EmbeddingRepository.getEmbedding(face_id=data['face_id'], embedding_id=embed.id)
            embeddings.append(embedding.embedding)
    if images is not None:
        len_images = len(data['images'])
        if len_images > 0:

            face = FaceRepository.getFace(user_id=data['user_id'], face_id=data['face_id'])
            embeddings = []
            uploaded_images = []

            # TODO: optimize this loop
            for image in images:
                # save file to storage
                prefix = str(data['user_id']) + '-update-face'
                uploaded_image = handle_upload(image, data=data, bucket='faces', prefix=prefix, storage=STORAGE)
                # save file to db
                file = FileRepository.createFile(uploaded_image, STORAGE)
                uploaded_images.append(file.id)
                # generate embeddings
                status, embed = img_to_encoding(uploaded_image, MODEL, resize=True)
                if not status:
                    logger.info(embed)
                    return create_response(status=422, message=embed)
                embeddings.append(embedding)
                keys.append(int(data['user_id']))
                values.append(embeddings)

            face_repo = FaceRepository()
            updated_face = face_repo.updateFaceEmbedding(
                face_id=data['face_id'],
                embeddings=embeddings,
                user_id=data['user_id'],
                files=uploaded_images)

    updated_face = updated_face.json
    return create_response(
        status='success',
        message="Successfully updated face {} with id: {}".format(updated_face.name, updated_face.face_id),
        data={
            'face': {
                'face_id': updated_face.face_id,
                'face_name': updated_face.face_name,
                'face_descr': updated_face.face_descr,
                'embeddings': serialize_embeddings(embeddings)
            }
        }
    )


# POST request for /face/recognize
@main.route("/face/recognize", methods=["POST"])
def recognize_face():
    data = dict(request.form)
    image = request.files['image']

    logger.info("Data recieved: %s", data)
    if "user_id" not in data:
        msg = "No user id provided for face recognition."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if not image:
        msg = "No image provided for face to recognize."
        logger.info(msg)
        return create_response(status=422, message=msg)

    faces = FaceRepository.getFaces(user_id=data['user_id'])

    database = {}
    keys = []
    values = []

    for face in faces:
        for embed in face.embeddings:
            embedding = EmbeddingRepository.getEmbedding(face_id=face.face_id, embedding_id=embed.id)
            database[face.face_id] = np.array(embedding.embedding).astype('float32')

            keys.append(int(data['user_id']))
            values.append(np.array(embedding.embedding).astype('float32'))

    # save file to storage
    prefix = str(data['user_id']) + '-recognize-face'
    uploaded_image = handle_upload(image, data=data, bucket='faces', prefix=prefix, storage=STORAGE)

    # Recognize image
    min_dist, identity, message, encoding = who_is_it_bulk(image_path=uploaded_image, model=MODEL,
                                                           user_id=data['user_id'])

    if message == "Face found":
        logger.info("Recognized identity id: %s with distance of: %s", identity, min_dist)
        face = FaceRepository.getFace(data['user_id'], face_id=int(identity))
        face = face.json

        # save file to db
        file = FileRepository.createFile(uploaded_image, STORAGE)

        # save encoding as recognized user
        face_repo = FaceRepository()
        updated_face = face_repo.updateFaceEmbedding(
            face_id=face['face_id'],
            embeddings=[encoding],
            user_id=data['user_id'],
            files=[file.id])

        # train new face
        loop.run_until_complete(train_knn(user_id=data['user_id']))

        return create_response(
            status=200,
            message=message,
            data={
                "face": {
                    "face_id": face['face_id'],
                    "face_name": face['face_name'],
                    "face_descr": face['face_descr'],
                    "accuracy": float((1 - min_dist) * 100)
                }
            }
        )
    else:
        logger.info("Cannot recognize face %s", min_dist)
        return create_response(
            status=200,
            message=message,
            data={}
        )


    # POST request for /face/recognize
@main.route("/face/verify", methods=["POST"])
def verify_face():
    data = dict(request.form)
    image = request.files['image']

    logger.info("Data recieved: %s", data)
    if "face_id" not in data:
        msg = "No face id provided for face you'd like to verify."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "user_id" not in data:
        msg = "No user id provided."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if not image:
        msg = "No image provided for face to verify."
        logger.info(msg)
        return create_response(status=422, message=msg)

    face = FaceRepository.getFace(face_id=data['face_id'], user_id=data['user_id'])

    database = {}

    for embedd in face.embeddings:
        embedding = EmbeddingRepository.getEmbedding(face_id=face.face_id, embedding_id=embedd.id)
        database[embedd.id] = np.array(embedding.embedding)

    # save file to storage
    prefix = str(data['user_id']) + '-verify-face'
    uploaded_image = handle_upload(image, data=data, bucket='faces', prefix=prefix, storage=STORAGE)
    # Verify face
    dist, match, encoding = verify(uploaded_image, database, MODEL)

    if match is False:
        logger.info("Un-Verified identity with distance of: %s", dist)
        return create_response(
            status=200,
            message='Image does not match face',
            data={}
        )
    if match is None:
        logger.info("Match is None.")
        return create_response(
            status=401,
            message=encoding,
            data={}
        )
    else:
        logger.info("Verified identity with distance of: %s", dist)
        face = FaceRepository.getFace(data['user_id'], face_id=data['face_id'])
        face = face.json

        # save file to db
        file = FileRepository.createFile(uploaded_image, STORAGE)

        # save encoding as recognized user
        face_repo = FaceRepository()
        updated_face = face_repo.updateFaceEmbedding(
            face_id=face['face_id'],
            embeddings=[encoding],
            user_id=data['user_id'],
            files=[file.id])

        # train new face
        loop.run_until_complete(train_knn(user_id=data['user_id']))

        return create_response(
            status=200,
            message='Image matches the face',
            data={
                "face": {
                    "face_id": face['face_id'],
                    "face_name": face['face_name'],
                    "face_descr": face['face_descr'],
                    "accuracy": float((1-dist)*100)
                }
            }
        )


@main.route("/face/detect", methods=["POST"])
def post_detect_face():
    data = dict(request.form)
    image = request.files['image']

    if not image:
        msg = "No image provided to detect face."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "user_id" not in data:
        msg = "No user id provided."
        logger.info(msg)
        return create_response(status=422, message=msg)

    # save file to storage
    prefix = str(data['user_id']) + '-detect-face'
    uploaded_image = handle_upload(image, data=data, bucket='faces', prefix=prefix, storage=STORAGE)
    logger.info(uploaded_image)

    logger.info('about to detect face')
    detected, faces, message, eyes = detect_face(uploaded_image)
    logger.info('detected face')

    if not detected:
        logger.info(message)
        return create_response(
            status=200,
            message=message,
            data={}
        )
    else:
        logger.info(message)
        logger.info(faces)
        logger.info(eyes)

        return create_response(
            status=200,
            message=message,
            data={
                "face": {
                    "no_faces": len(faces),
                    "face_coordinates": (faces.tolist()),
                    "eyes_coordinates": eyes
                }
            }
        )

    # commit it to database
    # db.session.add_all([new_person, email])
    # db.session.commit()


@main.route("/train-knn", methods=["GET"])
def train_knn_post():
    user_id = 1

    print("Training KNN...")
    loop.run_until_complete(train_knn(user_id=user_id))
    print("Training complete")

    return create_response(200, message="trainig done")


@main.route("/predict-knn", methods=["POST"])
def predict_knn_post():
    user_id = 1
    data = {}
    image = request.files['image']
    data['user_id'] = 1

    # print("Training KNN...")
    # loop.run_until_complete(train_knn(user_id=user_id))
    # print("Training complete")

    prefix = str(user_id) + '-predict-knn'
    uploaded_image = handle_upload(image, data=data, bucket='faces', prefix=prefix, storage=STORAGE)
    # save file to db
    file = FileRepository.createFile(file_url=uploaded_image, storage=STORAGE)
    # generate embeddings
    status, embed = img_to_encoding(uploaded_image, MODEL, resize=True)
    if not status:
        logger.info(embed)
        return create_response(status=422, message=embed)

    distance, pred, message = predict_face(embed, user_id)
    print(distance)
    print(pred)

    return create_response(200, data={
        "distance": distance,
        "pred": pred
    }, message=message)
