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

import numpy as np
from dotenv import load_dotenv
from comet_ml import API

# load environment variables defined in .env variables
load_dotenv('.env')

# Load environment variables. If not defied, default values are assigned
LOG_FILE = os.environ.get('FLASK_LOG', 'flask.log')
MODELS_DIR = os.getenv('MODELS_DIR', 'models/')
COMET_WORKSPACE = os.getenv('COMET_WORKSPACE')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL')
DEFAULT_VERSION = os.getenv('DEFAULT_VERSION')

 # a dictionary that maps each model to the features it uses: model->features (I had to do this after asking the TA on piazza)
features_map = {
    'final-best-task-6-xgb': [], # []: means the model uses all the features
    'xgboost-task5-model': ['distance', 'angle'],
    'lr-angle-model': ['angle'],
    'lr-distance-model': ['distance'],
    'lr-distance-angle-model': ['distance', 'angle']
}

# current model loaded in the server
curr_model = None
curr_model_name = None

# This function automatically retrieves the assets' information (filename) of a registered model from comet_ml
def get_registered_comet_model_file_name(workspace=COMET_WORKSPACE, model=DEFAULT_MODEL, version=DEFAULT_VERSION):
    api = API()
    try:
        model_details = api.get_registry_model_details(workspace, model, version)
        filename = model_details['assets'][0]['fileName']
        return filename
    except:
        return None

# Download and load a registered model from comet_ml
def get_registered_comet_model(filename, workspace=COMET_WORKSPACE, model=DEFAULT_MODEL, version=DEFAULT_VERSION, output_dir=MODELS_DIR):
    api = API()
    try:
        api.download_registry_model({workspace}, model, version, output_dir, expand=True)
        model = joblib.load(f'{output_dir}/{filename}')
        return model
    except:
        return None


# Logic of switching a model
def switch_model(workspace, model, version):
    # TODO: check to see if the model you are querying for is already downloaded
    filename = get_registered_comet_model_file_name(workspace, model, version)
    is_downloaded = os.path.isfile(f'{MODELS_DIR}/{filename}')

    status_code = 200
    
    global curr_model

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    if (is_downloaded):
        curr_model = joblib.load(f'{MODELS_DIR}/{filename}')
        app.logger.info('Model loaded from local repository')

    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    else:
        model = get_registered_comet_model(workspace=workspace, model=model, version=version, filename=filename)
        if model:
            curr_model = model
            app.logger.info('Model downloaded successfully from comet_ml')
        else:
            app.logger.info('Error occured while trying to download model comet_ml')
            status_code = 400

    # set current model name
    global curr_model_name
    curr_model_name = model

    return status_code



#### Server initialization + endpoints ####

app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # setup logging configuration: log file, log level, log format ...
    # logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', 
    #                     datefmt='%Y-%m-%d %H:%M:%S',
    #                     filename=LOG_FILE, 
    #                     level=logging.INFO, 
    #                     force=True)
    fileHandler = logging.FileHandler(LOG_FILE)
    fileHandler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(fileHandler)

    # Other initialization before the first request (load default model and start logging)
    app.logger.info('Flask app started')
    # get filename of the default model
    filename = get_registered_comet_model_file_name()
    # initialize current model and current model name
    global curr_model
    curr_model = get_registered_comet_model(filename)
    global curr_model_name
    curr_model_name = DEFAULT_MODEL
    app.logger.info('Default model loaded')


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    app.logger.info(f'API call: /logs')
    
    status_code = 200
    # read the log file
    with open(LOG_FILE, "r") as f:
        content = f.read()
  
    # return response and set status code
    return jsonify(content), status_code  # response must be json serializable!


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
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    app.logger.info(f'API call: /download_registry_model')
    json = request.get_json()
    app.logger.info(json)

    # get model information

    workspace = json['workspace']
    model = json['model']
    version = json['version']

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    status_code = switch_model(workspace, model, version)

    # Build and log response
    response = {
            'message': 'Model loaded successfully' if status_code==200
            else 'Error occured while trying to download model from comet_ml'
        }

    app.logger.info(response)

    # return response and set status code
    return jsonify(response), status_code  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Input JSON format:
    features (df.to_json())

    Returns predictions
    """
    # Get POST json data
    app.logger.info(f'API call: /predict')
    json = request.get_json()
    app.logger.info(json)

    global curr_model
    # features = np.array(json['features']).reshape(-1, 1)
    status_code = 200
    
    # feed model with features to generate prediction and log results
    try:
        df = pd.read_json(json)
        if df.empty:
            features = df[features_map[curr_model_name]]
            preds = curr_model.predict_proba(features)[:,1]
        else:
            preds = np.array([])
        app.logger.info(preds)
    except Exception as e:
        preds = np.array([])
        status_code = 400
        app.logger.info('Error occured in prediction')

    app.logger.info(preds)

    # return response and set status code
    return jsonify(preds.tolist()), status_code # response must be json serializable!
