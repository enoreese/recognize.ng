from flask import Blueprint, request
from api.models import db, Person, Email
from api.repositiory import ArtistRepository
from api.core import create_response, serialize_list, logger
from sqlalchemy import inspect
from flask import jsonify

main = Blueprint("artist", __name__)  # initialize blueprint


# function that is called when you visit /



# function that is called when you visit /persons
@main.route("/artist", methods=["GET"])
def get_person():
    artist_id = request.args.get('artist_id')

    logger.info("Data recieved: %s", artist_id)
    artist = ArtistRepository.getArtist(id=artist_id)
    artist = artist.json
    artist_obj = {
        "id": artist['id'],
        "artist_name": artist['artist_name'],
        "artist_description": artist['artist_description'],
        "record_label": artist['record_label'],
        # "songs": serialize_list(artist['songs'])
    }
    return create_response(status=200, data={"person": artist_obj})


# POST request for /persons
@main.route("/artist", methods=["POST"])
def create_person():
    data = request.get_json()

    logger.info("Data object: %s", data)
    if 'artist_name' not in data:
        msg = "No name provided for artist."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "artist_description" not in data:
        msg = "No description provided for artist."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "record_label" not in data:
        msg = "No record label provided for artist."
        logger.info(msg)
        return create_response(status=422, message=msg)

    print('here')

    # create SQLAlchemy Objects
    # try:
    new_artist = ArtistRepository.createArtist(
        artist_name=data['artist_name'],
        artist_description=data['artist_description'],
        record_label=data['record_label']
    )
    # except Exception as e:
    #     msg = e
    #     logger.info(msg)
    #     return create_response(status=422, message=msg)

    artist = new_artist.json
    artist_obj = {
        "id": artist['id'],
        "artist_name": artist['artist_name'],
        "artist_description": artist['artist_description'],
        "songs": serialize_list(artist['songs'])
    }

    logger.info("inserted user: %s", artist)
    # email = Email(email=data["email"])
    # new_person.emails.append(email)

    # commit it to database
    # db.session.add_all([new_person, email])
    # db.session.commit()
    return create_response(
        status=200,
        message="Successfully created person {} with id: {}".format(new_artist.artist_name, new_artist.id),
        data=artist_obj
    )


# POST request for /persons
# @main.route("/artist", methods=["PUT"])
# def update_person():
#     data = request.get_json()
#
#     logger.info("Data recieved: %s", data)
#     if "fullname" not in data:
#         msg = "No name provided for person."
#         logger.info(msg)
#         return create_response(status=422, message=msg)
#     if "email" not in data:
#         msg = "No email provided for person."
#         logger.info(msg)
#         return create_response(status=422, message=msg)
#     if "password" not in data:
#         msg = "No password provided for person."
#         logger.info(msg)
#         return create_response(status=422, message=msg)
#     if "phone" not in data:
#         msg = "No phone number provided for person."
#         logger.info(msg)
#         return create_response(status=422, message=msg)
#
#     # create SQLAlchemy Objects
#     new_person = PersonRepository.update(fullname=data["name"], email=data['email'], password=data['password'],
#                                          phone=data['phone'])
#     email = Email(email=data["email"])
#     new_person.emails.append(email)
#
#     return create_response(
#         status=200,
#         message="Successfully created person {new_person.name} with id: {new_person._id}",
#         # data=new_person
#     )
