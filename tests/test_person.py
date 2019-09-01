from api.models import db, Person
from api.repositiory import PersonRepository
import json

# client passed from client - look into pytest for more info about fixtures
# test client api: http://flask.pocoo.org/docs/1.0/api/#test-client

# def test_index(client):
#     rs = client.get("/person")
#     assert rs.status_code == 200


def test_get_person(client):
    # create Person and test whether it returns a person
    temp_person = PersonRepository.create(fullname="frank castle", password="12345", phone=7088289121, email='frank.castle@gmail.com')

    rs = client.get("/persons?phone=7088289121")
    ret_dict = rs.json
    print(rs)

    assert rs.status_code == 200
    # assert len(ret_dict["result"]["person"]) == 1
    assert ret_dict["result"]["person"]["fullname"] == "frank castle"

def test_post_person(client):
    rs = client.post(
        "/persons",
        content_type="application/json",
        data=json.dumps({
            "email": 'frank.castle@gmail.com',
            "fullname": 'frank castle',
            "password": '12345',
            "phone": 7088289121
        })
    )
    print (rs.json)
    assert rs.status_code == 200
    ret_dict = rs.json  # gives you a dictionary
    assert ret_dict["result"]["phone"] == 7088289121
