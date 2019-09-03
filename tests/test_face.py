from api.repositiory import PersonRepository, FaceRepository
import json, base64
from api.utils import img_to_encoding, create_model

MODEL = create_model()

# client passed from client - look into pytest for more info about fixtures
# test client api: http://flask.pocoo.org/docs/1.0/api/#test-client

# def test_index(client):
#     rs = client.get("/person")
#     assert rs.status_code == 200


def test_get_face(client):
    # create Person and test whether it returns a person
    temp_person = PersonRepository.create(fullname="frank castle", password="12345", phone=7088289121, email='frank.castle@gmail.com')
    person = temp_person.json
    temp_face = FaceRepository.createFace(
        face_name='pete castleone',
        face_descr='sweet petes face',
        user_id=person['id'],
        embeddings=[
            img_to_encoding(base64.encodebytes(open('/Users/sasu/Desktop/Dev/RecognizeNg/tests/test_images/frank.jpg', 'rb').read()).decode("utf-8"), MODEL)
        ]
    )
    temp_face = temp_face.json

    rs = client.get("/face?face_id={}&user_id={}".format(temp_face['face_id'], person['id']))
    ret_dict = rs.json

    assert rs.status_code == 200
    # assert len(ret_dict["result"]["person"]) == 1
    assert ret_dict["result"]["response"]["data"]['face_name'] == "pete castleone"

def test_post_face(client):
    temp_person = PersonRepository.create(fullname="frank castle", password="12345", phone=7088289121,
                                          email='frank.castle@gmail.com')
    person = temp_person.json
    rs = client.post(
        "/face",
        content_type="application/json",
        data=json.dumps({
            "user_id": person['id'],
            "face_name": 'pete castleone',
            "face_descr": 'sweet petes face',
            "images": [
                base64.encodebytes(open('/Users/sasu/Desktop/Dev/RecognizeNg/tests/test_images/frank.jpg', 'rb').read()).decode("utf-8")
            ]
        })
    )
    assert rs.status_code == 200
    ret_dict = rs.json  # gives you a dictionary
    assert ret_dict["result"]["person"]["fullname"] == "frank castle"
    assert ret_dict["result"]["face"]["face_name"] == "pete castleone"


def test_recognize_face(client):
    # Create temporary person
    temp_person = PersonRepository.create(fullname="frank castle", password="12345", phone=7088289121,
                                          email='frank.castle@gmail.com')
    person = temp_person.json

    # Create temporary face
    temp_face = FaceRepository.createFace(
        face_name='pete castleone',
        face_descr='sweet petes face',
        user_id=person['id'],
        embeddings=[
            img_to_encoding(base64.encodebytes(
                open('/Users/sasu/Desktop/Dev/RecognizeNg/tests/test_images/frank.jpg', 'rb').read()).decode("utf-8"),
                            MODEL)
        ]
    )
    temp_face = temp_face.json

    rs = client.post(
        "/face/recognize",
        content_type="application/json",
        data=json.dumps({
            "user_id": person['id'],
            "image":
                base64.encodebytes(
                    open('/Users/sasu/Desktop/Dev/RecognizeNg/tests/test_images/frank.jpg', 'rb').read()).decode(
                    "utf-8")
        })
    )


    assert rs.status_code == 200
    ret_dict = rs.json  # gives you a dictionary
    assert ret_dict["result"]["face"]["distance"] <= 0.7
    assert ret_dict["result"]["face"]["face_name"] == "pete castleone"


def test_verify_face(client):
    # Create temporary person
    temp_person = PersonRepository.create(fullname="frank castle", password="12345", phone=7088289121,
                                          email='frank.castle@gmail.com')
    person = temp_person.json

    # Create temporary face
    temp_face = FaceRepository.createFace(
        face_name='pete castleone',
        face_descr='sweet petes face',
        user_id=person['id'],
        embeddings=[
            img_to_encoding(base64.encodebytes(
                open('/Users/sasu/Desktop/Dev/RecognizeNg/tests/test_images/frank.jpg', 'rb').read()).decode("utf-8"),
                            MODEL)
        ]
    )
    temp_face = temp_face.json

    rs = client.post(
        "/face/verify",
        content_type="application/json",
        data=json.dumps({
            "user_id": person['id'],
            "face_id": temp_face['face_id'],
            "image":
                base64.encodebytes(
                    open('/Users/sasu/Desktop/Dev/RecognizeNg/tests/test_images/frank.jpg', 'rb').read()).decode(
                    "utf-8")
        })
    )


    assert rs.status_code == 200
    ret_dict = rs.json  # gives you a dictionary
    assert ret_dict["result"]["face"]["distance"] <= 0.7
    assert ret_dict["result"]["face"]["face_name"] == "pete castleone"

def test_detect_face(client):
    rs = client.post(
        "/face/detect",
        content_type="application/json",
        data=json.dumps({
            "image":
                base64.encodebytes(
                    open('/Users/sasu/Desktop/Dev/RecognizeNg/tests/test_images/frank.jpg', 'rb').read()).decode(
                    "utf-8")
        })
    )

    print(rs)

    assert rs.status_code == 200
    ret_dict = rs.json  # gives you a dictionary
    assert ret_dict["result"]["face"]["no_faces"] == 1
    # assert ret_dict["result"]["face"]["face_name"] == "pete castleone"