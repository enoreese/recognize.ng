from flask import Blueprint, request
from api.repositiory import PersonRepository, EmbeddingRepository, FingerprintRepository, SongRepository, ArtistRepository
from api.core import create_response, serialize_list, logger, serialize_embeddings
from api.utils.music import MusicUtils, recognize
import os

main = Blueprint("music", __name__)  # initialize blueprint


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in set(['mp3'])


# function that is called when you visit /persons
@main.route("/music", methods=["GET"])
def get_song():
    artist_id = request.args.get('artist_id')
    song_id = request.args.get('song_id')
    song = SongRepository.getSong(id=song_id, artist_id=artist_id)

    if song is None:
        msg = "No song found."
        logger.info(msg)
        return create_response(status=400, message=msg)

    logger.info("Face Object: %s", song.json)

    fingerprints = FingerprintRepository.getFingerprint(song_id=song.id)

    response = {
            "song_name": song.song_name,
            "song_id": song.id,
            "file_sha1": song.file_sha1.decode(),
            "fingerprinted": song.fingerprinted,
            "album": song.album,
            "fingerprints": fingerprints.song_hash.decode()
        }

    return create_response(status=200, data={"response": response})


# function that is called when you visit /persons
@main.route("/music/all", methods=["GET"])
def get_songs():
    songs = SongRepository.getAllSongs()

    logger.info("Songs Object: %s", songs)

    return create_response(status=200, data={"response": serialize_list(songs)})


# function that is called when you visit /song
@main.route("/music/artist/all", methods=["GET"])
def get_artist_songs():
    artist_id = request.args.get('artist_id')
    songs = SongRepository.getArtistSongs(artist_id=artist_id)

    logger.info("Faces Object: %s", songs)
    song_arr = []

    for song in songs:
        song_arr.append({
            "song_name": song.song_name,
            "song_id": song.id,
            "song_hash": song.file_sha1.decode()
        })

    return create_response(status=200, data={"response": song_arr})


# POST request for /song
@main.route("/music", methods=["POST"])
def create_song():
    data = dict(request.form)
    file = request.files['song']

    logger.info("Data recieved: %s", data)
    logger.info("File recieved: %s", file)
    if "artist_id" not in data:
        msg = "No artist id."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "song_name" not in data:
        msg = "No name provided for song."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if "album" not in data:
        msg = "No description provided for face."
        logger.info(msg)
        return create_response(status=422, message=msg)
    if not file and allowed_file(filename=file.filename):
        msg = "No description provided for face."
        logger.info(msg)
        return create_response(status=422, message=msg)

    filepath = os.path.join('api/uploads/', file.filename)
    file.save(filepath)
    logger.info("Saved File")

    artist = ArtistRepository.getArtist(id=data['artist_id'])
    logger.info("Get artist", artist.json)

    config = {
        "fingerprint_limit": 10
    }

    musicUtils = MusicUtils(config=config)
    logger.info("Get Music Utils")

    song, msg = musicUtils.fingerprint_file(
        artist_id=artist.id,
        album=data['album'],
        filepath=filepath,
        song_name=data['song_name']
    )

    logger.info("Song Object: %s", song.json)

    return create_response(
        status=200,
        message=msg,
        data={
            "song_name": song.song_name,
            "song_id": song.id,
            "artist_name": artist.artist_name,
            "file_sha1": song.file_sha1.decode(),
            "fingerprinted": song.fingerprinted,
            "album": song.album
        }
    )


# POST request for /face
@main.route("/recognize/song", methods=["POST"])
def recognize_song():
    data = request.get_json()
    files = request.files['song']

    logger.info("Data recieved: %s", data)
    logger.info("File recieved: %s", files)
    if not files:
        msg = "No audio file to recognize."
        logger.info(msg)
        return create_response(status=422, message=msg)

    config = {
        "fingerprint_limit": 10
    }

    musicUtils = MusicUtils(config=config)

    filepath = os.path.join('api/uploads/queries', files.filename)
    files.save(filepath)

    song = musicUtils.recognize(recognize.FileRecognizer, filepath)

    if song is None:
        msg = "Nothing recognized -- did you play the song out loud so your mic could hear it? :)"
        logger.info(msg)
        return create_response(status=301, message=msg)

    logger.info("Song Object: %s", song)

    return create_response(
        status=200,
        message="Successfully found song {} with id: {}".format(song['song_name'], song['song_id']),
        data=song
    )


