""" Defines the Song repository """

from api.models import Fingerprint, db
from sqlalchemy import func
import codecs

class FingerprintRepository():
    """ The repository for the user model """

    @staticmethod
    def getAllFingerprint():
        """ Query all Songs in DB """
        return Fingerprint.query.all()

    @staticmethod
    def getFingerprint(song_id):
        """ Query a Song by artist_id and song_id_id """
        return Fingerprint.query.filter_by(song_id=song_id).first()

    def searchHashes(self, split_values):
        """ Query a Song by artist_id and song_id_id """
        length = len(split_values)
        # print("split values", list(split_values))
        print("split values length", length)
        all_values = [value.upper() for value in split_values] * length
        print("all values length", len(all_values))
        print("all values [0]", all_values[-1])
        # count = 0
        # for split, all in zip(split_values, self.getAllFingerprint()):
        #     print("split", split)
        #     print('all ', all.song_hash.decode().upper())
        #     if split == all.song_hash.decode().upper():
        #         all_values.append(all.song_hash.decode().upper())
        #         count += 1
        # for values in split_values:
        #     all_values.append(values)
        # print("all values 2", len(all_values))
        result = db.session.query(Fingerprint).filter(Fingerprint.song_hash.in_(split_values)).all()
        print("result", result)
        return result

    @staticmethod
    def createFingerprint(split_values):
        '''

        :param song_hash:
        :param offset:
        :param song_id:
        :return:
        '''
        for values in split_values:
            fingerprint = Fingerprint(song_hash=values[0], offset=int(values[2]))

            fingerprint.song_id = values[1]

            db.session.add(fingerprint)

        # fingerprint.save()
        db.session.commit()

        return True
