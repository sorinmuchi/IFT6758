import os
from dotenv import load_dotenv
from tqdm.notebook import tqdm
import requests
import numpy as np
import pandas as pd



class NHL_API:
    
    def __init__(self):
        # Load dataset directory path from .env file
        load_dotenv('../.env')
        self.DATASET_DIR_PATH = os.getenv('DATASET_DIR_PATH')

        if (not self.DATASET_DIR_PATH):
            self.DATASET_DIR_PATH = 'dataset/'


    def get_nhl_data(self, start_season, end_season):
        """Return the NHL play-by-play data between two seasons.

        :param start_season: start season (start_season/start_season+1)
        :type a: int
        :param end_season: end season (end_season/end_season+1)
        :type b: int

        :rtype: list
        :return: NHL play-by-play data of all seasons between start_season and end_season
        """
        progress_bar = tqdm(range(start_season, end_season + 1))
        data = []
        for season in progress_bar:
            progress_bar.set_description(f'Retrieving NHL data for season {season}/{season + 1}')
            data.extend(self.get_nhl_season(season))
            progress_bar.set_description(f'Data saved to {self.DATASET_DIR_PATH}')

        return data


    def get_nhl_season(self, season):
        """Return the NHL play-by-play data for one season (regular season and playoffs)

        :param season: season (season/season+1)
        :type a: int

        :rtype: list
        :return: NHL play-by-play data of season/season+1
        """

        filepath = os.path.join(self.DATASET_DIR_PATH, f'data-{season}.npy')

        # check if the data already exists in local
        if os.path.exists(filepath):
            return np.load(filepath, allow_pickle=True)

        data = []
        data.extend(self.get_nhl_regular(season))
        data.extend(self.get_nhl_playoffs(season))

        # save the data in local for next usage     
        np.save(filepath, data)

        return data


    def get_nhl_regular(self, season, game_type = 2):
        """Retrieve and return the NHL play-by-play data for a regular season.

        :param season: season (season/season+1)
        :type a: int
        :game_type: type index of regular season in the NHL Stats API 
        :type a: int

        :rtype: list
        :return: NHL play-by-play data of regular season (season/season+1)
        """

        nb_games = 868 if season == 2020 else 1271 if season > 2016 else 1230
        data=[]

        for i in range(nb_games):
            game_id = f'{season}{game_type:02}{i+1:04}'
            url = f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/'
            response = requests.get(url)

            if(response.status_code != 200):
                raise Exception('Error occured while retrieving data from NHL api!')

            data.append(response.json())
        return data


    def get_nhl_playoffs(self, season, game_type = 3): 
        """Retrieve and return the NHL play-by-play data for the playoffs of season/season+1.

        :param season: season (season/season+1)
        :type a: int
        :game_type: type index of playoffs in the NHL Stats API 
        :type a: int

        :rtype: list
        :return: NHL play-by-play data of playoffs (season/season+1)
        """

        data=[]
        nb_rounds = 4
        nb_games = 7

        for iround in range(nb_rounds):
            for matchup in range(pow(2, 3 - iround)):
                for game in range(7):
                    game_id = f'{season}{game_type:02}0{iround+1}{matchup+1}{game+1}'
                    url = f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/'
                    response = requests.get(url)

                    if(response.status_code != 200):          
                        continue

                    data.append(response.json())
        return data
