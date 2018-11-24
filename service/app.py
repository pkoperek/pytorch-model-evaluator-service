import psycopg2
import torch
import logging

from flask import Flask, request, jsonify
app = Flask(__name__)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    request_as_json = request.get_json(force=True)
    log.debug(f'Request: {str(request_as_json)}')
    return jsonify(action='yolo')
