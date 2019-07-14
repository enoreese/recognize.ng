from api.models import Person

class PersonRepository:
    """ The repository for the user model """

    @staticmethod
    def get(email):
        """ Query a user by last and first name """
        return Person.query.filter_by(email=email).first()

    @staticmethod
    def getById(id):
        """ Query a user by last and first name """
        return Person.query.filter_by(id=id).first()

    def update(self, email, phone, password, fullname):
        """ Update a user's age """
        user = self.get(email)
        user.phone = phone

        return user.save()

    @staticmethod
    def create(fullname, password, phone, email):
        """ Create a new user """
        user = Person(email=email, phone=phone, fullname=fullname, password=password)

        return user.save()
