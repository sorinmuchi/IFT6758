
import requests
import pandas as pd
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)
def extractFeatures(fetchedData):

        fullGame = fetchedData
        game = fullGame['liveData']['plays']['allPlays']
        #################################################(1/20)
        # Remove scheduled games that did not take place
        if len(game) == 0:
            return np.nan
        #################################################(2/20)
        # Populate array of eventType
        eventType = []
        lastEventType = []
        lastEventPeriod = []
        lastEventPeriodTime = []
        lastEventXCoord = []
        lastEventYCoord = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                eventType.append(x['result']['event'])
                lastEventType.append(lastEvent['result']['event'])
                lastEventPeriod.append(lastEvent['about']['period'])
                lastEventPeriodTime.append(lastEvent['about']['periodTime'])
                if 'x' in lastEvent['coordinates']:
                    lastEventXCoord.append(lastEvent['coordinates']['x'])
                else:
                    lastEventXCoord.append(np.nan)
                if 'y' in lastEvent['coordinates']:
                    lastEventYCoord.append(lastEvent['coordinates']['y'])
                else:
                    lastEventYCoord.append(np.nan)
            lastEvent = x
        #################################################(3/20)
        # Populate array of Period
        period = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                period.append(x['about']['period'])
        #################################################(4/20)
        # Populate array of periodTime
        periodTime = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                periodTime.append(x['about']['periodTime'])
        #################################################(5/20)
        # Populate array of periodType
        periodType = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                periodType.append(x['about']['periodType'])
        #################################################(6/20)
        # Populate array of gameID
        gameID = [fullGame['gamePk']] * (len(periodType))
        #################################################(7/20)
        # Populate array of teamOfShooter
        teamOfShooter = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                teamOfShooter.append(x['team']['name'])
        #################################################(8/20)
        # Populate array of homeOrAway
        homeOrAway = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                if str(x['team']['name']) == str(fullGame['gameData']['teams']['away']['name']):
                    homeOrAway.append("away")
                if str(x['team']['name']) == str(fullGame['gameData']['teams']['home']['name']):
                    homeOrAway.append("home")
                    #################################################(9/20)
        # Populate arrays of x and y coordinates
        xCoord = []
        yCoord = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                if 'x' in x['coordinates']:
                    xCoord.append(x['coordinates']['x'])
                else:
                    xCoord.append(np.nan)
                if 'y' in x['coordinates']:
                    yCoord.append(x['coordinates']['y'])
                else:
                    yCoord.append(np.nan)
        #################################################(10/20)
        # Populate array of shooter
        shooter = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                shooter.append(x['players'][0]['player']['fullName'])
        #################################################(11/20)
        # Populate array of Goalie
        goalie = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                goalie.append(x['players'][len(x['players']) - 1]['player']['fullName'])
        #################################################(12/20)
        # Populate array of shotType
        shotType = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                if 'secondaryType' in x['result']:
                    shotType.append(x['result']['secondaryType'])
                else:
                    shotType.append(np.nan)
        #################################################(13/20)
        # Populate aray of emptyNet
        emptyNet = []
        for x in game:
            if str(x['result']['event']) == 'Shot':
                emptyNet.append(np.nan)
            if str(x['result']['event']) == 'Goal':
                if 'emptyNet' in x['result']:
                    emptyNet.append(x['result']['emptyNet'])
                else:
                    emptyNet.append(np.nan)
        #################################################(14/20)
        # Populate array of strength
        strength = []
        for x in game:
            if str(x['result']['event']) == 'Shot':
                strength.append(np.nan)
            if str(x['result']['event']) == 'Goal':
                strength.append(x['result']['strength']['name'])
        #################################################(15/20)
        # Populate array of season
        season = []
        for x in gameID:
            x = str(x)
            season.append(x[0:4])
        #################################################(16/20)
        # Populate array of rinkSide
        i = 0
        rinkSide = []
        for x in game:
            if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
                if str(periodType[i]) != "SHOOTOUT" and 'rinkSide' in \
                        fullGame['liveData']['linescore']['periods'][int(period[i]) - 1][str(homeOrAway[i])]:
                    info = fullGame['liveData']['linescore']['periods'][int(period[i]) - 1][str(homeOrAway[i])][
                        'rinkSide']
                    rinkSide.append(info)
                else:
                    rinkSide.append(np.nan)
                i += 1
        #################################################(17/20)
        # Populate array of gameType
        gameType = [fullGame['gameData']['game']['type']] * (len(periodType))
        #################################################(18/20)
        # Populate array of totalPlayTime
        totalPlayTime = []
        stringTotalPlayTime = ""
        i = 0
        for y in fullGame['liveData']['plays']['allPlays']:
            i += 1
            if i == len(fullGame['liveData']['plays']['allPlays']):
                # IMPLEMENTING THE ACTUAL TOTAL PLAYTIME
                if fullGame['gameData']['game']['type'] == "R":
                    if y['about']['period'] == 3:
                        stringTotalPlayTime = "60:00"
                    elif y['about']['period'] == 4:
                        extraTime = str(y['about']['periodTime'])
                        ex = extraTime.split(':')
                        minute = str(60 + int(ex[0]))
                        stringTotalPlayTime = minute + ":" + ex[1]
                    elif y['about']['period'] == 5:
                        stringTotalPlayTime = "65:00"
                if fullGame['gameData']['game']['type'] == "P":
                    if y['about']['period'] == 3:
                        stringTotalPlayTime = "60:00"
                    else:
                        prePeriod = int(y['about']['period']) - 1
                        minute = str(prePeriod * 2) + "0"
                        extraTime = str(y['about']['periodTime'])
                        ex = extraTime.split(':')
                        minute = str(int(minute) + int(ex[0]))
                        stringTotalPlayTime = minute + ":" + ex[1]

        dateformat = stringTotalPlayTime.split(':')

        if int(dateformat[0]) >= 60:
            if len(str(int(dateformat[0]) - 60)) == 2:

                if int(int(dateformat[0]) / 60) == 2:

                    if int(dateformat[0]) >= 120 and int(dateformat[0]) <= 129:
                        stringTotalPlayTime = "0" + str(int(int(dateformat[0]) / 60)) + ":0" + str(
                            int(dateformat[0]) - 120) + ":" + str(dateformat[1])
                    else:
                        stringTotalPlayTime = "0" + str(int(int(dateformat[0]) / 60)) + ":" + str(
                            int(dateformat[0]) - 120) + ":" + str(dateformat[1])
                else:

                    stringTotalPlayTime = "0" + str(int(int(dateformat[0]) / 60)) + ":" + str(
                        int(dateformat[0]) - 60) + ":" + str(dateformat[1])

            if len(str(int(dateformat[0]) - 60)) == 1:
                stringTotalPlayTime = "0" + str(int(int(dateformat[0]) / 60)) + ":0" + str(
                    int(dateformat[0]) - 60) + ":" + str(dateformat[1])
        else:
            stringTotalPlayTime = "00:" + stringTotalPlayTime

        totalPlayTime = [stringTotalPlayTime] * (len(periodType))
        #################################################(19/20)
        # Transform lists into series
        eventType = pd.Series(eventType)
        period = pd.Series(period)
        periodTime = pd.Series(periodTime)
        periodType = pd.Series(periodType)
        gameID = pd.Series(gameID)
        teamOfShooter = pd.Series(teamOfShooter)
        homeOrAway = pd.Series(homeOrAway)
        xCoord = pd.Series(xCoord)
        yCoord = pd.Series(yCoord)
        shooter = pd.Series(shooter)
        goalie = pd.Series(goalie)
        shotType = pd.Series(shotType)
        emptyNet = pd.Series(emptyNet)
        strength = pd.Series(strength)
        season = pd.Series(season)
        gameType = pd.Series(gameType)
        totalPlayTime = pd.Series(totalPlayTime)
        lastEventType = pd.Series(lastEventType)
        lastEventPeriod = pd.Series(lastEventPeriod)
        lastEventPeriodTime = pd.Series(lastEventPeriodTime)
        lastEventXCoord = pd.Series(lastEventXCoord)
        lastEventYCoord = pd.Series(lastEventYCoord)
        #################################################(20/20)
        # Make dataframe with series
        df = pd.DataFrame({'eventType': eventType, 'period': period, 'periodTime': periodTime, 'periodType': periodType,
                           'gameID': gameID,
                           'teamOfShooter': teamOfShooter, 'homeOrAway': homeOrAway, 'xCoord': xCoord, 'yCoord': yCoord,
                           'shooter': shooter, 'goalie': goalie,
                           'shotType': shotType, 'emptyNet': emptyNet, 'strength': strength, 'season': season,
                           'rinkSide': rinkSide, 'gameType': gameType, 'totalPlayTime': totalPlayTime,
                           'lastEventType': lastEventType, 'lastEventPeriod': lastEventPeriod,
                           'lastEventPeriodTime': lastEventPeriodTime, 'lastEventXCoord': lastEventXCoord,
                           'lastEventYCoord': lastEventYCoord})
        print(df)
        return df



class gameClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):

        logger.info(f"Initializing ClientGame; base URL: ")

    def pingGame(self, gameId="2021020329") :
        gameId= str(2021020329)
        fetchedData = requests.get("https://statsapi.web.nhl.com/api/v1/game/"+gameId+"/feed/live/")
        TididedData=extractFeatures(json.loads(fetchedData.text))
        print(TididedData)
        return TididedData

if __name__ == "__main__":

    Client=gameClient("127.0.0.1",5000)
    x=Client.pingGame()