from api.repositiory import SongRepository
from .decoder import unique_hash, path_to_songname, read
from .fingerprint import fingerprint, DEFAULT_WINDOW_SIZE, DEFAULT_FS, DEFAULT_OVERLAP_RATIO
from api.core import logger
import multiprocessing
import os
import traceback
import sys


class MusicUtils(object):

    SONG_ID = "song_id"
    SONG_NAME = 'song_name'
    CONFIDENCE = 'confidence'
    MATCH_TIME = 'match_time'
    OFFSET = 'offset'
    OFFSET_SECS = 'offset_seconds'

    def __init__(self, config):
        super(MusicUtils, self).__init__()

        self.config = config

        # if we should limit seconds fingerprinted,
        # None|-1 means use entire track
        self.limit = self.config.get("fingerprint_limit", None)
        if self.limit == -1:  # for JSON compatibility
            self.limit = None

        self.get_fingerprinted_songs()

    def get_fingerprinted_songs(self):
        # get songs previously indexed
        self.songs = SongRepository.getFingerprintedSongs() #self.db.get_songs()
        self.songhashes_set = set()  # to know which ones we've computed before
        for song in self.songs:
            song_hash = song.file_sha1
            self.songhashes_set.add(song_hash)

    # def fingerprint_directory(self, path, extensions, nprocesses=None):
    #     # Try to use the maximum amount of processes if not given.
    #     try:
    #         nprocesses = nprocesses or multiprocessing.cpu_count()
    #     except NotImplementedError:
    #         nprocesses = 1
    #     else:
    #         nprocesses = 1 if nprocesses <= 0 else nprocesses
    #
    #     pool = multiprocessing.Pool(nprocesses)
    #
    #     filenames_to_fingerprint = []
    #     for filename, _ in decoder.find_files(path, extensions):
    #
    #         # don't refingerprint already fingerprinted files
    #         if unique_hash(filename) in self.songhashes_set:
    #             print ("%s already fingerprinted, continuing..." % filename)
    #             continue
    #
    #         filenames_to_fingerprint.append(filename)
    #
    #     # Prepare _fingerprint_worker input
    #     worker_input = zip(filenames_to_fingerprint,
    #                        [self.limit] * len(filenames_to_fingerprint))
    #
    #     # Send off our tasks
    #     iterator = pool.imap_unordered(_fingerprint_worker,
    #                                    worker_input)
    #
    #     # Loop till we have all of them
    #     while True:
    #         try:
    #             song_name, hashes, file_hash = iterator.next()
    #         except multiprocessing.TimeoutError:
    #             continue
    #         except StopIteration:
    #             break
    #         except:
    #             print("Failed fingerprinting")
    #             # Print traceback because we can't reraise it here
    #             traceback.print_exc(file=sys.stdout)
    #         else:
    #             sid = self.db.insert_song(song_name, file_hash)
    #
    #             self.db.insert_hashes(sid, hashes)
    #             self.db.set_song_fingerprinted(sid)
    #             self.get_fingerprinted_songs()
    #
    #     pool.close()
    #     pool.join()

    def fingerprint_file(self, artist_id, album, filepath, song_name=None):
        # songname = path_to_songname(filepath)
        song_hash = unique_hash(filepath)
        song_name = song_name

        # don't refingerprint already fingerprinted files
        if song_hash in self.songhashes_set:
            msg = "%s already fingerprinted, continuing..." % song_name
            logger.info(msg)
            song = SongRepository.getSongByHash(hash=song_hash)
            return song, msg
        else:
            song_name, hashes, file_hash = _fingerprint_worker(
                filepath,
                self.limit,
                song_name=song_name
            )


            song = SongRepository.createSong(song_name=song_name, artist_id=artist_id, album=album,
                                             file_hash=file_hash, hashes=hashes)

            # sid = self.db.insert_song(song_name, file_hash)
            # self.db.insert_hashes(sid, hashes)
            song_repo = SongRepository()
            song_ = song_repo.setSongFingerprinted(song_id=song.id, artist_id=artist_id)

            # self.db.set_song_fingerprinted(sid)
            self.get_fingerprinted_songs()

            msg = "%s successfully fingerprinted..." % song_name
            logger.info(msg)

            return song_, msg

    def find_matches(self, samples, Fs=DEFAULT_FS):
        hashes = fingerprint(samples, Fs=Fs)
        matches = SongRepository.returnMatches(hashes=hashes)
        # return self.db.return_matches(hashes)
        return matches

    def align_matches(self, matches):
        """
            Finds hash matches that align in time with other matches and finds
            consensus about which hashes are "true" signal from the audio.
            Returns a dictionary with match information.
        """
        # align by diffs
        diff_counter = {}
        largest = 0
        largest_count = 0
        song_id = -1
        for tup in matches:
            print("tup", tup)
            sid, diff = tup
            if diff not in diff_counter:
                diff_counter[diff] = {}
            if sid not in diff_counter[diff]:
                diff_counter[diff][sid] = 0
            diff_counter[diff][sid] += 1

            if diff_counter[diff][sid] > largest_count:
                largest = diff
                largest_count = diff_counter[diff][sid]
                song_id = sid

        # extract idenfication
        # song = self.db.get_song_by_id(song_id)
        song = SongRepository.getSongById(id=song_id)
        if song:
            # TODO: Clarify what `get_song_by_id` should return.
            songname = song.song_name
        else:
            return None

        # return match info
        nseconds = round(float(largest) / DEFAULT_FS *
                         DEFAULT_WINDOW_SIZE *
                         DEFAULT_OVERLAP_RATIO, 5)
        song = {
            MusicUtils.SONG_ID: song_id,
            MusicUtils.SONG_NAME: songname,
            MusicUtils.CONFIDENCE: largest_count,
            MusicUtils.OFFSET: int(largest),
            MusicUtils.OFFSET_SECS: nseconds,
            'file_sha1': song.file_sha1.decode(),
        }
        return song

    def recognize(self, recognizer, *options, **kwoptions):
        r = recognizer(self)
        return r.recognize(*options, **kwoptions)


def _fingerprint_worker(filename, limit=None, song_name=None):
    # Pool.imap sends arguments as tuples so we have to unpack
    # them ourself.
    try:
        filename, limit = filename
    except ValueError:
        pass

    # songname, extension = os.path.splitext(os.path.basename(filename))
    song_name = song_name
    channels, Fs, file_hash = read(filename, limit)
    print(channels)
    print(file_hash)
    result = set()
    channel_amount = len(channels)

    for channeln, channel in enumerate(channels):
        # TODO: Remove prints or change them into optional logging.
        print("Fingerprinting channel %d/%d for %s" % (channeln + 1,
                                                       channel_amount,
                                                       filename))
        hashes = fingerprint(channel, Fs=Fs)
        print("Finished channel %d/%d for %s" % (channeln + 1, channel_amount,
                                                 filename))
        print("hashes", hashes)
        result |= set(hashes)

    return song_name, result, file_hash


def chunkify(lst, n):
    """
    Splits a list into roughly n equal parts.
    http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
    """
    return [lst[i::n] for i in range(n)]