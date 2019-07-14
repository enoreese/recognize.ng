from flask import Blueprint, request
from api.models import db, Person, Email
from api.repositiory import PersonRepository
from api.core import create_response, serialize_list, logger
from sqlalchemy import inspect

main = Blueprint("person", __name__)  # initialize blueprint


# function that is called when you visit /
@main.route("/person")
def index():
    # you are now in the current application context with the main.route decorator
    # access the logger with the logger from api.core and uses the standard logging module
    # try using ipdb here :) you can inject yourself
    logger.info("Hello World!")
    return "<h1>Hello World!</h1>"


# function that is called when you visit /persons
@main.route("/persons/<email>", methods=["GET"])
def get_person():
    email = request.view_args['email']
    persons = PersonRepository.get(email=email)
    return create_response(data={"person": serialize_list(persons)})


# POST request for /persons
@main.route("/persons", methods=["POST"])
def create_person():
    data = request.get_json()

    logger.info("Data recieved: %s", data)
    if "fullname" not in data:
        msg = "No name provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "email" not in data:
        msg = "No email provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "password" not in data:
        msg = "No password provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "phone" not in data:
        msg = "No phone number provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)

    # create SQLAlchemy Objects
    new_person = PersonRepository.create(fullname=data["name"], email=data['email'], password=data['password'],
                                         phone=data['phone'])
    email = Email(email=data["email"])
    new_person.emails.append(email)

    # commit it to database
    db.session.add_all([new_person, email])
    db.session.commit()
    return create_response(
        status='success',
        message= "Successfully created person {new_person.name} with id: {new_person._id}"
    )


# POST request for /persons
@main.route("/persons", methods=["PUT"])
def update_person():
    data = request.get_json()

    logger.info("Data recieved: %s", data)
    if "fullname" not in data:
        msg = "No name provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "email" not in data:
        msg = "No email provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "password" not in data:
        msg = "No password provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "phone" not in data:
        msg = "No phone number provided for person."
        logger.info(msg)
        return create_response(status=422, message=msg)

    # create SQLAlchemy Objects
    new_person = PersonRepository.create(fullname=data["name"], email=data['email'], password=data['password'],
                                         phone=data['phone'])
    email = Email(email=data["email"])
    new_person.emails.append(email)

    # commit it to database
    db.session.add_all([new_person, email])
    db.session.commit()
    return create_response(
        status='success',
        message= "Successfully created person {new_person.name} with id: {new_person._id}"
    )
