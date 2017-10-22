# coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import pandas as pd
import numpy as np
import requests
import time
from slackclient import SlackClient
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
# NLP Spacy
import spacy
nlp = spacy.load('en')
# Mongo 
from pymongo import MongoClient
client = MongoClient('localhost', 27007)
db = client.chatbot
collection = db.botmemory

# Fix Python 2.x.
try: 
    input = raw_input
except NameError: 
    pass

def clean_record():
    record = {
        "input": {
            "user": "__default__",
            "request": None
        },
        "output": {
            "text": None
        },
        "feedback": {
            "reward": 0
        }
    }
    return record

def compute_weather(place):
    """
    Compute the weather (temperature and status)
    using a given location

    arg:
        place: str 

    return:
        text: str, phrase of the found condition
    """
    iconmap = {
        "01": ":sunny:",
        "02": ":partly_sunny:",
        "03": ":partly_sunny:",
        "04": ":cloud:",
        "09": ":droplet:",
        "10": ":droplet:",
        "11": ":zap:",
        "13": ":snowflake:",
        "50": ":umbrella:",    # mist?
    }

    weather_api_key = os.environ.get("WEATHER_API_KEY")
    if not weather_api_key:
        return "Please set as the WEATHER_API_KEY environment variable to a " \
            "valid (free) OpenWeatherMap API key: " \
            "http://openweathermap.org/appid#get"
    url = 'http://api.openweathermap.org/data/2.5/forecast/daily?'\
        'q={0}&cnt=5&mode=json&units=metric&APPID={1}'.format(
            place, weather_api_key)

    dat = requests.get(url).json()

    msg = ["{0}: ".format(dat["city"]["name"])]
    for day in dat["list"]:
        name = time.strftime("%a", time.gmtime(day["dt"]))
        high = str(int(round(float(day["temp"]["max"]))))
        icon = iconmap.get(day["weather"][0]["icon"][:2], ":question:")
        msg.append(u"{0} {1}° {2}".format(name, high, icon))

    return " ".join(msg)


def memorize(record):
    record["date"] = time.strftime('%m/%d-%H:%M')
    #record_id = collection.insert_one(record).inserted_id
    collection.insert(record)
    #return record_id


def ask_reward():
    reward = None
    input_user = input('is my answer OK ?')
    if input_user == 'yes':
        reward = 1
    elif input_user == 'no':
        reward = -1
    return reward

def ask_reward_slack(text, channel):
    reward = None
    text='is my answer OK ?'
    slack_client.api_call("chat.postMessage", channel=channel, text=text)
    #input_user = input('is my answer OK ?')
    #if input_user == 'yes':
    #    reward = 1
    #elif input_user == 'no':
    #    reward = -1
    #return reward

def answer(text, channel, BASH, record):
    """
    decide what to do with the answer depending on the mode of the bot
    """
    if BASH == True:
        record['output']['text'] = str(text)
        answer_bash(text)
        reward = ask_reward()
        record['feedback']['reward'] = reward
        print(record)
        if reward:
            memorize(record)
    else:
        answer_slack(text, channel)

def answer_bash(text):
    """
    send the answer to the command line
    """
    print(text)


def answer_slack(text, channel):
    """
    send the answer to slack 
    """
    slack_client.api_call("chat.postMessage", channel=channel, text=text)


def parse_entities(command):
    """
    Parse entities for a given phrase
    """
    print(command)
    doc = nlp(command)
    print(doc, command, doc.ents)
    if(doc.ents):
        response = 'Entities: '
        for ent in doc.ents:
            response += str(ent.text) + ", "
    else:
        response = "There are no entities in the phrase"
    print(response)
    return response


def available_velibs(command):
    """
    find the available number of velibs for a given station
    """
    response = requests.get("http://opendata.paris.fr/api/records/1.0/download/?dataset=stations-velib-disponibilites-en-temps-reel&facet=banking&facet=bonus&facet=status&facet=contract_name&rows=-1")
    txt = response.text
    f = open('data/velib.csv', 'w+')
    f.write(txt)
    velibs = pd.read_csv('data/velib.csv', sep=";")
    velibs = velibs[velibs.status == 'OPEN']
    subdf = velibs[velibs['name'].str.contains(command.upper())]
    resp = ''
    for num,row in subdf.iterrows():
        if row['status'] == 'OPEN':
            status_stand = ':white_check_mark: '
        elif row['status'] == 'CLOSED':
            status_stand = ':x: '

        resp = resp + status_stand +  str(row['address']) + ', there are ' + str(row['available_bikes']) + ' :bike: and ' + str(row['available_bike_stands']) + ' :parking:\n'
    #resp = np.array_str(subdf[["name","address","available_bikes"]].values)
    return resp
