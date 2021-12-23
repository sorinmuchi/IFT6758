# -*- coding: utf-8 -*-
import os

import requests
import pandas as pd
import logging
import numpy as np
import json
import json
logger = logging.getLogger(__name__)

if not (os.path.isfile('tracked.json') and os.access('tracked.json', os.R_OK)):
    with open('tracked.json', 'w') as outfile:
        data = {}
        json.dump(data, outfile)


def getTeams(gameId):
    fetchedData = requests.get("https://statsapi.web.nhl.com/api/v1/game/" + gameId + "/feed/live/")
    game =json.loads(fetchedData.text)['liveData']['plays']['allPlays']
    teamOfShooter = set()
    for x in game:
        if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
            teamOfShooter.add(x['team']['name'])

    return teamOfShooter




def extractFeatures(fetchedData,gameId,team_Shooter,idx=0):
    try:
        with open('tracked.json') as f:
            data = json.load(f)
        idx=int(data[gameId][team_Shooter])
    except:
        idx=0

    fullGame=fetchedData
    game = fullGame['liveData']['plays']['allPlays']
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
    ii = 0
    for x in game:
        if ii == 0:
            lastEvent = x
        if str(x['result']['event']) == 'Shot' or str(x['result']['event']) == 'Goal':
            eventType.append(x['result']['event'])
            if ii != 0:
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
        ii += 1
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
                info = fullGame['liveData']['linescore']['periods'][int(period[i]) - 1][str(homeOrAway[i])]['rinkSide']
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
    stringTotalPlayTime = "20:00"
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
    df = pd.DataFrame(
        {'eventType': eventType, 'period': period, 'periodTime': periodTime, 'periodType': periodType, 'gameID': gameID,
         'teamOfShooter': teamOfShooter, 'homeOrAway': homeOrAway, 'xCoord': xCoord, 'yCoord': yCoord,
         'shooter': shooter, 'goalie': goalie,
         'shotType': shotType, 'emptyNet': emptyNet, 'strength': strength, 'season': season, 'rinkSide': rinkSide,
         'gameType': gameType, 'totalPlayTime': totalPlayTime,
         'lastEventType': lastEventType, 'lastEventPeriod': lastEventPeriod, 'lastEventPeriodTime': lastEventPeriodTime,
         'lastEventXCoord': lastEventXCoord, 'lastEventYCoord': lastEventYCoord})



    df2 = df


    penaltyGameSeconds = []
    penaltyLength = []
    penaltyOver = []
    penaltyTeam = []

    a = fetchedData["liveData"]["plays"]["penaltyPlays"]
    c = fetchedData["liveData"]["plays"]["allPlays"]

    period1 = []
    periodTime1 = []

    for x in a:
        period1.append(c[x]["about"]["period"])
        periodTime1.append(c[x]["about"]["periodTime"])
        penaltyLength.append(float(c[x]["result"]["penaltyMinutes"]) * 60)
        penaltyTeam.append(c[x]['team']['name'])
        penaltyOver.append(False)

    for p, pt in zip(period1, periodTime1):
        pt = pt.split(":")
        ##############################################
        if p == 1:
            time = 60 * int(pt[0]) + int(pt[1])
        if p == 2:
            time = 60 * 20 + 60 * int(pt[0]) + int(pt[1])
        if p == 3:
            time = 60 * 40 + 60 * int(pt[0]) + int(pt[1])
        if p == 4:
            time = 60 * 60 + 60 * int(pt[0]) + int(pt[1])
        if p == 5:
            time = 60 * 80 + 60 * int(pt[0]) + int(pt[1])
        if p == 6:
            time = 60 * 100 + 60 * int(pt[0]) + int(pt[1])
        if p == 7:
            time = 60 * 120 + 60 * int(pt[0]) + int(pt[1])
        if p == 8:
            time = 60 * 140 + 60 * int(pt[0]) + int(pt[1])
        if p == 9:
            time = 60 * 160 + 60 * int(pt[0]) + int(pt[1])
        if p == 10:
            time = 60 * 180 + 60 * int(pt[0]) + int(pt[1])
        if p == 11:
            time = 60 * 200 + 60 * int(pt[0]) + int(pt[1])
        penaltyGameSeconds.append(time)

    Goal = []
    EmptyNet = []
    distance = []
    angle = []
    gameSeconds = []
    lastEventGameSeconds = []
    timeFromLastEvent = []
    distanceFromLastEvent = []
    rebound = []
    lastEventAngle = []
    changeInAngleShot = []
    speed = []

    for x in df2["eventType"]:
        if x == "Shot":
            Goal.append(0)
        if x == "Goal":
            Goal.append(1)

    for x in df2["emptyNet"]:
        if x == False or pd.isna(x):
            EmptyNet.append(0)
        elif x == True:
            EmptyNet.append(1)

    rNet = [89, 0]
    lNet = [-89, 0]

    for r, x, y, x2, y2, e in zip(df2["rinkSide"], df2["xCoord"], df2["yCoord"], df2["lastEventXCoord"],
                                  df2["lastEventYCoord"], df2["lastEventType"]):
        if str(e) == "Shot":
            rebound.append(True)
        else:
            rebound.append(False)

        if r == "right":
            d = np.sqrt((lNet[0] - x) ** 2 + (lNet[1] - y) ** 2)
            distance.append(d)
            if y < 0:
                a = 180 - np.degrees(np.arcsin(np.abs(lNet[0] - x) / d))
                angle.append(a)
                if str(e) == "Shot":
                    d2 = np.sqrt((lNet[0] - x2) ** 2 + (lNet[1] - y2) ** 2)
                    if y2 < 0:
                        a2 = 180 - np.degrees(np.arcsin(np.abs(lNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                    if y2 > 0:
                        a2 = np.degrees(np.arcsin(np.abs(lNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                else:
                    lastEventAngle.append(np.nan)
            elif y > 0:
                a = np.degrees(np.arcsin(np.abs(lNet[0] - x) / d))
                angle.append(a)
                if str(e) == "Shot":
                    d2 = np.sqrt((lNet[0] - x2) ** 2 + (lNet[1] - y2) ** 2)
                    if y2 < 0:
                        a2 = 180 - np.degrees(np.arcsin(np.abs(lNet[0] - x2) / d2))
                        lastEventAngle.append(a2)

                    if y2 > 0:
                        a2 = np.degrees(np.arcsin(np.abs(lNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                else:
                    lastEventAngle.append(np.nan)
            else:
                angle.append(90)
                if str(e) == "Shot":
                    d2 = np.sqrt((lNet[0] - x2) ** 2 + (lNet[1] - y2) ** 2)
                    if y2 < 0:
                        a2 = 180 - np.degrees(np.arcsin(np.abs(lNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                    if y2 > 0:
                        a2 = np.degrees(np.arcsin(np.abs(lNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                else:
                    lastEventAngle.append(np.nan)
        elif r == "left":
            d = np.sqrt((rNet[0] - x) ** 2 + (rNet[1] - y) ** 2)
            distance.append(d)
            if y < 0:
                a = np.degrees(np.arcsin(np.abs(rNet[0] - x) / d))
                angle.append(a)
                if str(e) == "Shot":
                    d2 = np.sqrt((rNet[0] - x2) ** 2 + (rNet[1] - y2) ** 2)
                    if y2 < 0:
                        a2 = np.degrees(np.arcsin(np.abs(rNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                    if y2 > 0:
                        a2 = 180 - np.degrees(np.arcsin(np.abs(rNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                else:
                    lastEventAngle.append(np.nan)
            elif y > 0:
                a = 180 - np.degrees(np.arcsin(np.abs(rNet[0] - x) / d))
                angle.append(a)
                if str(e) == "Shot":
                    d2 = np.sqrt((rNet[0] - x2) ** 2 + (rNet[1] - y2) ** 2)
                    if y2 < 0:
                        a2 = np.degrees(np.arcsin(np.abs(rNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                    if y2 > 0:
                        a2 = 180 - np.degrees(np.arcsin(np.abs(rNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                else:
                    lastEventAngle.append(np.nan)
            else:
                angle.append(90)
                if str(e) == "Shot":
                    d2 = np.sqrt((rNet[0] - x2) ** 2 + (rNet[1] - y2) ** 2)
                    if y2 < 0:
                        a2 = np.degrees(np.arcsin(np.abs(rNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                    if y2 > 0:
                        a2 = 180 - np.degrees(np.arcsin(np.abs(rNet[0] - x2) / d2))
                        lastEventAngle.append(a2)
                else:
                    lastEventAngle.append(np.nan)
        elif pd.isna(r):
            lastEventAngle.append(np.nan)
            if x > 0:
                d = np.sqrt((rNet[0] - x) ** 2 + (rNet[1] - y) ** 2)
                distance.append(d)
                if y < 0:
                    a = np.degrees(np.arcsin(np.abs(rNet[0] - x) / d))
                    angle.append(a)
                elif y > 0:
                    a = 180 - np.degrees(np.arcsin(np.abs(rNet[0] - x) / d))
                    angle.append(a)
                else:
                    angle.append(90)

            else:
                d = np.sqrt((lNet[0] - x) ** 2 + (lNet[1] - y) ** 2)
                distance.append(d)
                if y < 0:
                    a = 180 - np.degrees(np.arcsin(np.abs(lNet[0] - x) / d))
                    angle.append(a)
                elif y > 0:
                    a = np.degrees(np.arcsin(np.abs(lNet[0] - x) / d))
                    angle.append(a)
                else:
                    angle.append(90)
    i = 0
    for p, pt, lp, lpt in zip(df2["period"], df2["periodTime"], df2["lastEventPeriod"], df2["lastEventPeriodTime"]):
        pt = pt.split(":")
        lpt = lpt.split(":")
        ##############################################
        if p == 1:
            time = 60 * int(pt[0]) + int(pt[1])
        if p == 2:
            time = 60 * 20 + 60 * int(pt[0]) + int(pt[1])
        if p == 3:
            time = 60 * 40 + 60 * int(pt[0]) + int(pt[1])
        if p == 4:
            time = 60 * 60 + 60 * int(pt[0]) + int(pt[1])
        if p == 5:
            time = 60 * 80 + 60 * int(pt[0]) + int(pt[1])
        if p == 6:
            time = 60 * 100 + 60 * int(pt[0]) + int(pt[1])
        if p == 7:
            time = 60 * 120 + 60 * int(pt[0]) + int(pt[1])
        if p == 8:
            time = 60 * 140 + 60 * int(pt[0]) + int(pt[1])
        if p == 9:
            time = 60 * 160 + 60 * int(pt[0]) + int(pt[1])
        if p == 10:
            time = 60 * 180 + 60 * int(pt[0]) + int(pt[1])
        if p == 11:
            time = 60 * 200 + 60 * int(pt[0]) + int(pt[1])
        ##############################################
        if lp == 1:
            time2 = 60 * int(lpt[0]) + int(lpt[1])
        if lp == 2:
            time2 = 60 * 20 + 60 * int(lpt[0]) + int(lpt[1])
        if lp == 3:
            time2 = 60 * 40 + 60 * int(lpt[0]) + int(lpt[1])
        if lp == 4:
            time2 = 60 * 60 + 60 * int(lpt[0]) + int(lpt[1])
        if p == 5:
            time2 = 60 * 80 + 60 * int(pt[0]) + int(pt[1])
        if p == 6:
            time2 = 60 * 100 + 60 * int(pt[0]) + int(pt[1])
        if p == 7:
            time2 = 60 * 120 + 60 * int(pt[0]) + int(pt[1])
        if p == 8:
            time2 = 60 * 140 + 60 * int(pt[0]) + int(pt[1])
        if p == 9:
            time = 60 * 160 + 60 * int(pt[0]) + int(pt[1])
        if p == 10:
            time = 60 * 180 + 60 * int(pt[0]) + int(pt[1])
        if p == 11:
            time = 60 * 200 + 60 * int(pt[0]) + int(pt[1])

        gameSeconds.append(time)
        lastEventGameSeconds.append(time2)
        timeFromLastEvent.append(time - time2)

    for x, y, x2, y2 in zip(df2["xCoord"], df2["yCoord"], df2["lastEventXCoord"], df2["lastEventYCoord"]):
        if pd.isna(x2) or pd.isna(y2):
            distanceFromLastEvent.append(np.nan)
        else:
            d = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
            distanceFromLastEvent.append(d)

    for a1, a2 in zip(angle, lastEventAngle):
        if pd.isna(a1) or pd.isna(a2):
            changeInAngleShot.append(np.nan)
        else:
            changeInAngleShot.append(np.abs(a1 - a2))

    for d, t in zip(distanceFromLastEvent, timeFromLastEvent):
        if pd.isna(d) or pd.isna(t) or d == 0 or t == 0:
            speed.append(np.nan)
        else:
            speed.append(float(d) / float(t))

    timeSincePP = []
    numFriendlySkaters = []
    numOpposingSkaters = []
    strength2 = []
    #####################################################################################################################################################################
    for g, team, event in zip(gameSeconds, df2["teamOfShooter"], df2["eventType"]):

        penG1 = []
        penL1 = []
        penO1 = []
        penG2 = []
        penL2 = []
        penO2 = []
        fskaters = 5
        oskaters = 5
        for a, b, c, d in zip(penaltyGameSeconds, penaltyLength, penaltyOver, penaltyTeam):
            if d == team:
                penG1.append(a)
                penL1.append(b)
                penO1.append(c)

        value = 0
        i = 0
        for pg, po, pl in zip(penG1, penO1, penL1):

            # if a goal happens, set po=True unless pl==5
            # set po=True any if g>pg for b
            if po == True:
                i += 1
                continue
            if g > pg:
                value = g - pg
                fskaters -= 1
                if (g - pg) >= pl:
                    fskaters += 1
                    if i + 1 < len(penG1):
                        if not g > penG1[i + 1]:
                            value = 0
                            penO1[i] = True
                        else:
                            if (g - penG1[i + 1]) >= penL1[i + 1]:
                                value = 0
                                penO1[i] = True
                                penO1[i + 1] = True
                            else:
                                fskaters -= 1
                    else:
                        value = 0
                        penaltyOver[i] = True
                else:
                    if i + 1 < len(penG1):
                        if g > penG1[i + 1]:
                            fskaters -= 1
                k = 0
                l = 0
                for t, p in zip(penaltyTeam, penaltyOver):
                    if t == team:
                        penaltyOver[l] = penO1[k]
                        k += 1
                    l += 1
                break
            else:
                break
        for a, b, c, d in zip(penaltyGameSeconds, penaltyLength, penaltyOver, penaltyTeam):
            if d != team:
                penG2.append(a)
                penL2.append(b)
                penO2.append(c)
        value2 = 0
        i = 0
        for pg, po, pl in zip(penG2, penO2, penL2):
            # if a goal happens, set po=True unless pl==5
            # set po=True any if g>pg for b
            if po == True:
                i += 1
                continue
            if g > pg:
                value2 = g - pg
                oskaters -= 1
                if (g - pg) >= pl:
                    oskaters += 1
                    if i + 1 < len(penG2):
                        if not g > penG2[i + 1]:
                            value2 = 0
                            penO2[i] = True
                        else:
                            if (g - penG2[i + 1]) >= penL2[i + 1]:
                                value2 = 0
                                penO2[i] = True
                                penO2[i + 1] = True
                            else:
                                oskaters -= 1
                    else:
                        value2 = 0
                        penO2[i] = True
                else:
                    if i + 1 < len(penG2):
                        if g > penG2[i + 1]:
                            oskaters -= 1
                if event == "Goal":
                    if pl != 5:
                        penO2[i] = True
                    if i + 1 < len(penG2):
                        if (g - penG2[i + 1]) < penL2[i + 1]:
                            if penL2 != 300.0:
                                penO2[i + 1] = True

                k = 0
                l = 0
                for t, p in zip(penaltyTeam, penaltyOver):
                    if t != team:
                        penaltyOver[l] = penO2[k]
                        k += 1
                    l += 1
                break
            else:
                break
        timeSincePP.append(value2)
        numFriendlySkaters.append(fskaters)
        numOpposingSkaters.append(oskaters)
        if fskaters == oskaters:
            st = "Even"
            strength2.append(st)
        if fskaters > oskaters:
            st = "Power Play"
            strength2.append(st)
        if fskaters < oskaters:
            st = "Short Handed"
            strength2.append(st)

    Goal = pd.Series(Goal)
    EmptyNet = pd.Series(EmptyNet)
    distance = pd.Series(distance)
    angle = pd.Series(angle)
    gameSeconds = pd.Series(gameSeconds)
    lastEventGameSeconds = pd.Series(lastEventGameSeconds)
    timeFromLastEvent = pd.Series(timeFromLastEvent)
    distanceFromLastEvent = pd.Series(distanceFromLastEvent)
    rebound = pd.Series(rebound)
    lastEventAngle = pd.Series(lastEventAngle)
    changeInAngleShot = pd.Series(changeInAngleShot)
    speed = pd.Series(speed)
    timeSincePP = pd.Series(timeSincePP)
    numFriendlySkaters = pd.Series(numFriendlySkaters)
    numOpposingSkaters = pd.Series(numOpposingSkaters)
    strength2 = pd.Series(strength2)

    dic = {"Goal": Goal, "EmptyNet": EmptyNet, "distanceFromNet": distance, "angle": angle,
           "gameSeconds": gameSeconds, "lastEventGameSeconds": lastEventGameSeconds,
           "timeFromLastEvent": timeFromLastEvent, "distanceFromLastEvent": distanceFromLastEvent,
           "rebound": rebound, "lastEventAngle": lastEventAngle, "changeInAngleShot": changeInAngleShot,
           "speed": speed, "timeSincePowerPlayStarted": timeSincePP,
           "numFriendlyNonGoalieSkaters": numFriendlySkaters, "numOpposingNonGoalieSkaters": numOpposingSkaters,
           "strength2": strength2}
    dfToJoin = pd.DataFrame(dic)




    dfOut = pd.concat([df2, dfToJoin], axis=1)

    dfOut['Goal'] = dfOut['Goal'].astype(np.int64)

    dfOut = dfOut.rename({'Goal': 'is_goal', 'distanceFromNet': 'distance'}, axis=1)
    dfOut=dfOut[dfOut['teamOfShooter']==str(team_Shooter)]
    dfOut = dfOut.reset_index(drop=True)
    if not  (dfOut.empty ):
        lastLine=dfOut.iloc[-1:].index.values[0]


        f = open('tracked.json')
        data = json.load(f)


        if (str(gameId) in data.keys() ) and (team_Shooter in data[gameId].keys()):

            data[gameId][team_Shooter]=str(lastLine)

        elif str(gameId) in data.keys():

            data[gameId][team_Shooter] =str(lastLine)
        else :
            data.update({gameId: {team_Shooter: str(lastLine)}})

        with open('tracked.json', 'w') as outfile:
            json.dump(data, outfile)

        return dfOut[idx+1:]
    return dfOut





class gameClient:
    def __init__(self):

        logger.info(f"Initializing ClientGame; base URL: ")

    def pingGame(self, team,gameId="2021020329",idx=0) :
        fetchedData = requests.get("https://statsapi.web.nhl.com/api/v1/game/"+gameId+"/feed/live/")
        TididedData=extractFeatures(json.loads(fetchedData.text),gameId,team,idx=idx)

        return TididedData

# if __name__ == "__main__":

    # Client=gameClient("127.0.0.1",5000)
    # team="Washington Capitals"
    # team1="Carolina Hurricanes"
    # print(getTeams("2021020329"))
    # x=Client.pingGame(team)
    # from serving_client import AppClient
    # serving = AppClient("127.0.0.1",5000)
    # res = serving.predict(x, "2021020329",team)
    # print(res)
