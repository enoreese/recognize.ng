from api.models import Person,db, Email

class PersonRepository:
    """ The repository for the user model """

    @staticmethod
    def get(user_id):
        """ Query a user by last and first name """
        return Person.query.filter_by(id=user_id).first()

    @staticmethod
    def getById(id):
        """ Query a user by last and first name """
        return Person.query.filter_by(id=id).first()

    def update(self, email, phone, password, fullname):
        """ Update a user's age """
        user = self.get(email)
        user.phone = phone

        db.session.add(user)
        db.session.commit()

        return user

    @staticmethod
    def create(fullname, password, phone, email):
        """ Create a new user """
        user = Person(phone=phone, fullname=fullname, password=password)

        print('user')

        email = Email(email=email)
        user.emails.append(email)

        print('email to save')

        email.save()
        user.save()

        print('saved')

        return user
