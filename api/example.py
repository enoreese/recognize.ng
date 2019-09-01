import warnings
import json
warnings.filterwarnings("ignore")
from api import create_app
import testing.postgresql
import time
from api.utils.music import MusicUtils
from api.utils.music.recognize import MicrophoneRecognizer

# load config from a JSON file (or anything outputting a python dictionary)
# with open("dejavu.cnf.SAMPLE") as f:
#     config = json.load(f)

def postgres():
    """
    The postgres Fixture. Starts a postgres instance inside a temp directory
    and closes it after tests are done.
    """
    with testing.postgresql.Postgresql() as postgresql:
        yield postgresql


# We spin up a temporary postgres instance
# in which we inject it into the app
def client(postgres):
    config_dict = {
        "SQLALCHEMY_DATABASE_URI": postgres.url(),
        "DEBUG": True,
        "TESTING": True,
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
    }
    app = create_app(config_dict)
    app.app_context().push()

    time.sleep(2)
    from api.models import db

    # app.run(host="0.0.0.0", port=5050)

    db.create_all()
    # for test client api reference
    # http://flask.pocoo.org/docs/1.0/api/#test-client
    client = app.test_client()
    yield client

if __name__ == '__main__':
	config = {
		"fingerprint_limit": 10
	}

	client = client(postgres)

	# create a Dejavu instance
	djv = MusicUtils(config)

	# Fingerprint all the mp3's in the directory we give it
	djv.fingerprint_directory("mp3", [".mp3"])

	# Recognize audio from a file
	# song = djv.recognize(FileRecognizer, "mp3/Sean-Fournier--Falling-For-You.mp3")
	# print "From file we recognized: %s\n" % song

	# Or recognize audio from your microphone for `secs` seconds
	secs = 5
	song = djv.recognize(MicrophoneRecognizer, seconds=secs)
	if song is None:
		print ("Nothing recognized -- did you play the song out loud so your mic could hear it? :)")
	else:
		print ("From mic with %d seconds we recognized: %s\n" % (secs, song))

	# Or use a recognizer without the shortcut, in anyway you would like
	# recognizer = FileRecognizer(djv)
	# song = recognizer.recognize_file("mp3/Josh-Woodward--I-Want-To-Destroy-Something-Beautiful.mp3")
	# print "No shortcut, we recognized: %s\n" % song