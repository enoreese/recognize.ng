""" Defines the File repository """

from api.models import Files


class FileRepository():
    """ The repository for the file model """

    @staticmethod
    def getFiles():
        """ Query all Files """
        return Files.query.all()

    @staticmethod
    def getFile(id):
        """ Query a File by id """
        return Files.query.filter_by(id=id).first()

    @staticmethod
    def createFile(file_url, storage):
        """ Create a new face embedding file """
        file = Files(
            file_url=file_url,
            storage=storage
        )

        file.save()

        return file
