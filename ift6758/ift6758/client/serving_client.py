import json
import os

import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class AppClient:

    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        if not (os.path.isfile('predicted.json') and os.access('predicted.json', os.R_OK)):
            with open('predicted.json', 'w') as outfile:
                data = {}
                json.dump(data, outfile)
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")



        # any other potential initialization

    def predict(self, X: pd.DataFrame,gameId) -> list:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
            gameId: to track the already predicted events
        """
        data= {
            "features": X.to_json(),
        }
        response_API = requests.post(self.base_url+"/predict",data=data)
        prediction=json.loads(response_API.text)["prediction"]



        f = open('predicted.json')
        data = json.load(f)
        if str(gameId) in data.keys():
            lastValue=eval(data[gameId])
            lastValue.append(prediction)
            d1 = {gameId: str(lastValue)}
            data.update(d1)
        with open('predicted.json', 'w') as outfile:
            json.dump(data, outfile)

        return eval(data[gameId])

        #raise NotImplementedError("TODO: implement this function")


    def logs(self) -> str:
        """Get server logs"""
        response_API =  requests.get(self.base_url + "/logs")
        return json.loads(response_API.text)["content"]


        #raise NotImplementedError("TODO: implement this function")

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        data={
            'workspace':workspace,
            'model': model,
            'version':version,
            'filename':'DownloadedModel'
        }
        response_API = requests.post(self.base_url+"/download_registry_model",data=data)
        return json.loads(response_API.text)

        #raise NotImplementedError("TODO: implement this function")

if __name__ == "__main__":

    Client=AppClient("127.0.0.1",5000)