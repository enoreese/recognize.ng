""" Defines the Song repository """

from api.models import Song, Fingerprint
from api.core import grouper
from .Fingerprint import FingerprintRepository
from .Artist import ArtistRepository


class SongRepository():
    """ The repository for the user model """

    @staticmethod
    def getAllSongs():
        """ Query all Songs in DB """
        return Song.query.all()

    @staticmethod
    def getSong(id, artist_id):
        """ Query a Song by artist_id and song_id_id """
        return Song.query.filter_by(id=id, artist=artist_id).first()

    @staticmethod
    def getSongById(id):
        """ Query a Song by artist_id and song_id_id """
        return Song.query.filter_by(id=id).first()

    @staticmethod
    def getSongByHash(hash):
        """ Query a Song by artist_id and song_id_id """
        return Song.query.filter_by(file_sha1=hash).first()

    @staticmethod
    def getArtistSongs(artist_id):
        """ Query all Songs by artist_id """
        return Song.query.filter_by(artist=artist_id).all()

    @staticmethod
    def getFingerprintedSongs():
        """ Query all fingerprinted Songs """
        return Song.query.filter_by(fingerprinted=1).all()

    @staticmethod
    def returnMatches(hashes):
        """ Query all fingerprinted Songs """
        # Create a dictionary of hash => offset pairs for later lookups
        mapper = {}
        for hash, offset in hashes:
            mapper[hash.upper()] = offset

        # Get an iteratable of all the hashes we need
        values = mapper.keys()

        for split_values in grouper(values, 1000):
            val = list(split_values)
            # print("split values", val)
            print_repo = FingerprintRepository()
            hashes = print_repo.searchHashes(split_values=val)
            print("hashes from hashes", hashes)

            for hash in hashes:
                print("hash in hashes ", hash)
                yield (hash.song_id, hash.offset - mapper[hash.song_hash.decode().upper()])

    def setSongFingerprinted(self, song_id, artist_id):
        '''Query to set song to fingerprinted'''
        song = self.getSong(id=song_id,artist_id=artist_id)

        song.fingerprinted = 1

        song.save()
        return song

    @staticmethod
    def createSong(song_name, album, artist_id, file_hash, hashes):
        '''

        :param song_name:
        :param artist_id:
        :return: song
        '''
        song = Song(song_name=song_name, album=album, file_sha1=file_hash)
        song.save()

        values = []
        for hash, offset in hashes:
            values.append((hash, song.id, offset))

        for split_values in grouper(values, 1000):
            fingerprint = FingerprintRepository.createFingerprint(split_values=(split_values))


        artist_repo = ArtistRepository()
        artist = artist_repo.addArtistSong(artist_id=artist_id, song=song)

        song.artist = artist.id
        song.save()

        return song
