import tensorflow as tf
import numpy as np
import cv2
from fr_utils import *
from inception_blocks_v2 import *


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by the formula

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = (pos_dist - neg_dist) + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


def verify(image_path, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    ### START CODE HERE ###

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    dist = 0

    # Step 2: Compute distance with identity's image (≈ 1 line)
    for (name, db_enc) in database.items():

        dist += np.linalg.norm(db_enc - encoding)

    final_dist = dist / len(database)

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if final_dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        match = True
    else:
        print("It's not " + str(identity) + ", please go away")
        match = False

    ### END CODE HERE ###

    return dist, match

def gray_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    imgtest1 = img.copy()
    imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)

    return imgtest

def detect_eyes(roi_gray):
    eye_cascade = cv2.CascadeClassifier('haar_classifiers/eye_detect_model.xml')

    eyes = eye_cascade.detectMultiScale(roi_gray)

    return eyes


def detect_face(image_file):
    imgtest = gray_image(image_file=image_file)
    face_cascade = cv2.CascadeClassifier('haar_classifier/face_detect_model.xml')

    faces = face_cascade.detectMultiScale(imgtest, scaleFactor=1.2, minNeighbors=5)

    no_faces = len(faces)

    if no_faces == 0:
        detected = False
        message = "No face has been detected"
    else:
        if no_faces == 1:
            detected = True
            message = "A face has been detected"
        else:
            detected = True
            message = "Multiple faces detected"

        eyes = []
        for (x, y, w, h) in faces:
            roi_gray = imgtest[y:y + h, x:x + w]

            eyes.append(detect_eyes(roi_gray=roi_gray))


    return detected, faces, message, eyes


def who_is_it(image_path, database, model):
    """
    Implements face recognition function for finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ## Step 1: Compute the target "encoding" for the image.
    encoding = img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
        message = 'No face found'
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        message = "Face found"

    return min_dist, identity, message


def create_model():
    print('Building CNN Architecture...')
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))

    print("Total Params:", FRmodel.count_params())

    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(FRmodel)

    return FRmodel