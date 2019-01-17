import psycopg2
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import time

from flask import g
from flask import Flask, request, jsonify
app = Flask(__name__)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm1d(32)
        self.head = nn.Linear(32 * 222, 3)

    def forward(self, x):
        app.logger.debug("Network input: " + str(x.size()))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


current_model_id = -1
model = DQN().to('cpu')


def create_connection():
    hostname = os.getenv('POSTGRES_HOST', 'localhost')
    port = int(os.getenv('POSTGRES_PORT', '5432'))
    username = os.getenv('POSTGRES_USERNAME', 'samm')
    password = os.getenv('POSTGRES_PASSWORD', 'samm_secret')
    dbname = os.getenv('POSTGRES_DATABASE', 'samm_db')

    return psycopg2.connect(
        host=hostname,
        port=port,
        user=username,
        password=password,
        dbname=dbname
    )


def get_db():
    if not hasattr(g, 'psql_db'):
        g.psql_db = create_connection()
        g.psql_cursor = g.psql_db.cursor()
    return g.psql_cursor


@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'psql_db'):
        g.psql_cursor.close()
        g.psql_db.close()


def model_with_max_score():
    global current_model_id
    db = get_db()

    db.execute(
        '''
        SELECT model_id
        FROM model_evaluations
        WHERE evaluation_score = (
            SELECT MAX(evaluation_score)
            FROM model_evaluations
        )
        '''
    )

    first_row = db.fetchone()
    if first_row:
        return first_row[0]

    return current_model_id


def best_model_in_db():
    global current_model_id

    db = get_db()

    now = time.time()
    db.execute(
        '''
            SELECT
                liv.evaluation_score AS live_score,
                sim.evaluation_score AS sim_score,
                sim.model_id AS sim_model
            FROM (
                SELECT
                    evaluation_score,
                    evaluation_end
                FROM model_evaluations
                WHERE
                    evaluation_type = 'LIVE'
                    AND model_id = %s
                    AND to_timestamp( evaluation_end )::date
                        = to_timestamp( %s )::date
            ) liv
            JOIN
            (
                SELECT
                    model_id,
                    evaluation_score,
                    evaluation_end
                FROM model_evaluations
                WHERE
                    evaluation_type = 'SIMULATION'
                    AND to_timestamp( evaluation_end )::date
                        = to_timestamp( %s )::date
            ) sim
            ON sim.evaluation_end = liv.evaluation_end
            ORDER BY liv.evaluation_end DESC
            LIMIT 10
        ''',
        (current_model_id, now, now)
    )

    comparison = db.fetchall()

    if len(comparison) > 0:
        first_row = comparison[0]
        live_score = first_row[0]
        sim_score = first_row[1]
        challenger_model_id = first_row[2]

        if live_score < sim_score:
            return challenger_model_id
    else:
        app.logger.warning(f'No data retrieved for {current_model_id}')

    return current_model_id


def retrieve_model_parameters(model_id):
    db = get_db()

    db.execute(
        '''
        SELECT
            model_data
        FROM models
        WHERE model_id = %s
        ''',
        (model_id, )
    )

    row = db.fetchone()

    if row:
        return row[0]
    else:
        raise RuntimeError(f'Fetch received 0 rows for model: {model_id}')


def update_model(new_model_id):
    global model
    global current_model_id

    app.logger.info(f'Updating the model to {new_model_id}')
    model_parameters = retrieve_model_parameters(new_model_id)
    model.load_state_dict(model_parameters)
    model.eval()
    current_model_id = new_model_id


def evaluate_environment_state(metrics):
    global model
    state = torch.tensor(metrics)
    with torch.no_grad():
        return model(state).max(1)[1].view(1, 1).item()


def evaluate_request(request_as_json):
    global current_model_id

    if current_model_id == -1:
        app.logger.debug('No model present - ask to do nothing')
        model_response = 0
    else:
        app.logger.debug('Evaluating the NN')
        metrics = request_as_json['metrics']
        model_response = evaluate_environment_state(metrics)

    return {
        'action': model_response,
        'model_version': str(current_model_id)
    }


@app.route('/evaluate', methods=['POST'])
def evaluate():
    global current_model_id

    request_as_json = request.get_json(force=True)
    app.logger.debug(f'Request: {str(request_as_json)}')

    if current_model_id == -1:
        app.logger.info('No model present... Selecting model with max score')
        best_model_id = model_with_max_score()
    else:
        app.logger.info('Looking for a better model')
        best_model_id = best_model_in_db()

    app.logger.debug(f'''
        Best model in db: {best_model_id} (current: {current_model_id})
    '''.strip())

    if best_model_id != current_model_id:
        app.logger.debug('Updating the model...')
        update_model(best_model_id)

    model_response = evaluate_request(request_as_json)

    app.logger.debug(
        f"The model has chosen action: {model_response['action']}"
    )

    return jsonify(model_response)
