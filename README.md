## What?

A simple service which enables evaluating a model for input coming from a
request coming in a form of JSON request.

## Dev setup

* Create virtual env (`direnv` will set it up on its own)
* Install required libraries with `pip3 install -r requirements.txt`
* Run the app in debug mode: `FLASK_DEBUG=1 FLASK_APP=service/app.py flask run`

Sample query:

`curl -X POST -d '{}' localhost:5000/evaluate`
