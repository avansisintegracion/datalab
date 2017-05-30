# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import pandas as pd
import numpy as np
import requests
from slackclient import SlackClient
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
import pyowm
owm = pyowm.OWM('7491e475f58f9fab017349742ee03997')  
import spacy
nlp = spacy.load('en')

def compute_weather(place):
    """
    Compute the weather (temperature and status)
    using a given location

    arg:
        place: str 

    return:
        text: str, phrase of the found condition
    """
    observation = owm.weather_at_place(place)
    w = observation.get_weather()
    temperature = w.get_temperature('celsius')['temp']
    text = "It is " + str(w.get_detailed_status()) + \
           " and it is " + str(temperature) + "C " + \
           "in " + observation.get_location().get_name()
    return text


def memorize(record):
    record["date"] = datetime.datetime.utcnow()
    record_id = collection.insert_one(record).inserted_id
    return record_id


def ask_reward():
    reward = None
    input_user = raw_input('is my answer OK ?')
    if input_user == 'yes':
        reward = 1
    elif input_user == 'no':
        reward = -1
    return reward


def answer_bash(text):
    """
    send the answer to the command line
    """
    print(text)


def answer_on_slack(text, channel):
    """
    send the answer to slack 
    """
    slack_client.api_call("chat.postMessage", channel=channel, text=text)


def answer(text, channel, BASH, record):
    """
    decide what to do with the answer depending on the mode of the bot
    """
    if BASH == True:
        record['output']['text'] = text
        answer_bash(text)
        reward = ask_reward()
        record['feedback']['reward'] = reward
        if reward:
            memorize(record)
    else:
        answer_on_slack(text, channel)


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
    f = open('velib.csv', 'w+')
    f.write(txt)
    velibs = pd.read_csv('velib.csv', sep=";")
    velibs = velibs[velibs.status == 'OPEN']
    subdf = velibs[velibs['name'].str.contains(command.upper())]
    resp = np.array_str(subdf[["name","address","available_bikes"]].values)
    return resp
