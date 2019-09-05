#!/bin/bash
# setup.sh 
# Setup script for Flask Boilerplate only for Mac machines. Look at docs for windows

set -o errexit  # exit on any errors

sudo apt install python3.7

sudo apt install postgresql postgresql-contrib

#brew install python3
#brew upgrade python3
#brew install postgresql
#brew link postgresql
#brew services start postgresql
pip install virtualenv
virtualenv flask-venv
source flask-venv/bin/activate
pip install -r requirements-2.txt

# wait until postgres is started
while ! pg_isready -h "localhost" -p "5432" > /dev/null 2> /dev/null; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 3
done

>&2 echo "Postgres is up - executing command"

createdb || true    # create init database - pass on error     # pass on error
psql -c "ALTER USER forge WITH SUPERUSER;" || true
psql -c "create database recognize owner forge encoding 'utf-8';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE recognize TO forge;"

python manage.py recreate_db
