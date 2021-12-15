"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib

import ift6758

import datetime
import numpy as np
from dotenv import load_dotenv
from comet_ml import API

load_dotenv('.env')
LOG_FILE = os.environ.get('FLASK_LOG', 'flask.log')
MODELS_DIR = os.getenv('MODELS_DIR', 'models/')
COMET_WORKSPACE = os.getenv('COMET_WORKSPACE')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL')
DEFAULT_VERSION = os.getenv('DEFAULT_VERSION')
DEFAULT_FILENAME = os.getenv('DEFAULT_FILENAME')
loaded_model = None


def get_registered_comet_model(workspace=COMET_WORKSPACE, model=DEFAULT_MODEL, version=DEFAULT_VERSION, output_dir=MODELS_DIR, filename=DEFAULT_FILENAME):
    api = API()
    try:
        api.download_registry_model({workspace}, model, version, output_dir, expand=True)
        model = joblib.load(f'{output_dir}/{filename}')
        return model
    except:
        return None



app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=LOG_FILE, 
                        level=logging.INFO, 
                        force=True)

    # TODO: any other initialization before the first request (e.g. load default model)
    app.logger.info('Flask app started')
    global loaded_model
    loaded_model = get_registered_comet_model()
    app.logger.info('Default model loaded')


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    # Get POST json data
    app.logger.info(f'API call: /logs')
    
    # TODO: read the log file specified and return the data
    # raise NotImplementedError("TODO: implement this endpoint")
    with open(LOG_FILE, "r") as f:
        content = f.read()

    print(type(content))
    print(content)

    response = {
            'type': 'Success',
            'content': str(content)
        }
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            filename: (required)
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    app.logger.info(f'API call: /download_registry_model')
    json = request.get_json()
    app.logger.info(json)

    workspace = json['workspace']
    model = json['model']
    version = json['version']
    filename = json['filename']

    # TODO: check to see if the model you are querying for is already downloaded
    is_downloaded = os.path.isfile(f'{MODELS_DIR}/{filename}')

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    status_code = 200
    global loaded_model

    if (is_downloaded):
        loaded_model = joblib.load(f'{MODELS_DIR}/{filename}')
        app.logger.info('Model loaded from local repository')

    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    else:
        model = get_registered_comet_model(workspace=workspace, model=model, version=version, filename=filename)
        if model:
            loaded_model = model
            app.logger.info('Model downloaded successfully from comet_ml')
        else:
            app.logger.info('Error occured while trying to download model comet_ml')
            status_code = 400

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    response = {
            'type': 'Success' if status_code==200 else 'Error',
            'content': 'Model loaded successfully' if status_code==200
            else 'Error occured while trying to download model from comet_ml.'
        }

    app.logger.info(response)
    return jsonify(response), status_code  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Input JSON format:
    {
        features: (required)
    }

    Returns predictions
    """
    # Get POST json data
    app.logger.info(f'API call: /predict')
    json = request.get_json()
    app.logger.info(json)

    global loaded_model
    features = np.array(json['features']).reshape(-1, 1)
    status_code = 200
    try:
        preds = loaded_model.predict(features)
        app.logger.info(preds)
    except Exception as e:
        preds = None
        status_code = 400
        app.logger.info('Error occured in prediction')
    print(preds)

    # TODO:
    # raise NotImplementedError("TODO: implement this enpdoint")
    response = {
            'type': 'Success' if status_code==200 else 'Error',
            'prediction': preds.tolist()
        }

    app.logger.info(response)
    return jsonify(response), status_code # response must be json serializable!
