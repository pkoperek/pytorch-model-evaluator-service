import psycopg2
import torch

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    request_as_json = request.get_json(force=True)
    app.logger.debug(f'Request: {str(request_as_json)}')
    return jsonify(action='yolo')
