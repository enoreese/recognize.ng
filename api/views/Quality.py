from flask import Blueprint, request
from api.core import create_response, logger
from api.utils import handle_upload, url_to_image

import requests, json
import numpy as np
from keras.preprocessing import image

main = Blueprint("quality", __name__)  # initialize blueprint

STORAGE = 'cloudinary'


# function that is called when you visit /
@main.route("/quality")
def index():
    # you are now in the current application context with the main.route decorator
    # access the logger with the logger from api.core and uses the standard logging module
    # try using ipdb here :) you can inject yourself
    logger.info("Hello Quality!")
    return "<h1>Hello Quality!</h1>"

# POST request for /id-quality
@main.route("/id-quality", methods=["POST"])
def id_quality():
    data = request.get_json()
    imag = request.files['image']

    logger.info("Data recieved: %s", data)
    if not imag:
        msg = "No image provided for quality prediction."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "user_id" not in data:
        msg = "No name provided for face."
        logger.info(msg)
        return create_response(status=422, message=msg)

    # save file to storage
    prefix = str(data['user_id']) + '-id-quality'
    uploaded_image = handle_upload(imag, data=data, bucket='quality', prefix=prefix, storage=STORAGE)

    img1 = url_to_image(uploaded_image)
    # decoded_image = decode_image(data['image'])
    #
    # img = BytesIO(decoded_image)

    img = image.img_to_array(image.load_img(img1,
                                            target_size=(150, 150))) / 255.

    # this line is added because of a bug in tf_serving < 1.11
    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:9000/v1/models/image_quality_classifier:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    pred = (np.array(pred['predictions'])[0] > 0.4).astype(np.int)
    if pred == 0:
        prediction = 'Bad'
    else:
        prediction = 'Good'

    # indicate that the request was a success

    return create_response(
        status=200,
        message="Successfully predicted image quality for your image",
        data={
            'prediction': prediction
        }
    )


# POST request for /id-quality
@main.route("/id-quality-bulk", methods=["POST"])
def id_quality_bulk():
    data = request.get_json()
    images = request.files.getlist("images")

    logger.info("Data recieved: %s", data)
    if not images:
        msg = "No image provided for quality prediction."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "user_id" not in data:
        msg = "No name provided for face."
        logger.info(msg)
        return create_response(status=422, message=msg)

    if len(images) < 0:
        msg = "Image array is empty."
        logger.info(msg)
        return create_response(status=422, message=msg)

    count = 1
    images_arr = {}

    for image_file in images:

        # img = BytesIO(image_file.read())
        # save file to storage
        prefix = str(data['user_id']) + '-id-quality'
        uploaded_image = handle_upload(image_file, data=data, bucket='quality', prefix=prefix, storage=STORAGE)

        img1 = url_to_image(uploaded_image)

        img = image.img_to_array(image.load_img(img1,
                                                target_size=(150, 150))) / 255.

        # this line is added because of a bug in tf_serving < 1.11
        img = img.astype('float16')

        # Creating payload for TensorFlow serving request
        payload = {
            "instances": [{'input_image': img.tolist()}]
        }

        # Making POST request
        r = requests.post('http://localhost:9000/v1/models/image_quality_classifier:predict', json=payload)

        # Decoding results from TensorFlow Serving server
        pred = json.loads(r.content.decode('utf-8'))

        pred = (np.array(pred['predictions'])[0] > 0.4).astype(np.int)
        if pred == 0:
            prediction = 'Bad'
        else:
            prediction = 'Good'

        images_arr[count] = prediction

        count += 1

    # indicate that the request was a success

    return create_response(
        status=200,
        message="Successfully predicted image quality for your image",
        data={
            'predictions': images_arr
        }
    )


