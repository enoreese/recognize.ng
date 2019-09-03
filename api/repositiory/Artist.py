""" Defines the Song repository """

from api.models import Artist

class ArtistRepository():
    """ The repository for the user model """

    @staticmethod
    def getAllArtists():
        """ Query all Songs in DB """
        return Artist.query.all()

    @staticmethod
    def getArtist(id):
        """ Query a Song by artist_id and song_id_id """
        return Artist.query.filter_by(id=id).first()

    def addArtistSong(self, artist_id, song):
        '''Update Artist Songs if not does not exist'''
        artist = self.getArtist(id=artist_id)

        artist.songs.append(song)

        print('done appending song')

        artist.save()

        print('done saving artist')

        return artist

    @staticmethod
    def createArtist(artist_name, artist_description, record_label):
        '''

        :param song_name:
        :param artist_id:
        :return: song
        '''
        fingerprint = Artist(artist_name=artist_name, artist_description=artist_description, record_label=record_label)

        fingerprint.save()

        return fingerprint
