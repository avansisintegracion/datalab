# coding: utf-8
import sys
import os
import time
import json
import requests
import pandas as pd
from slackclient import SlackClient
from chatterbot import ChatBot
import spacy
from pymongo import MongoClient
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Interpreter
reload(sys)
sys.setdefaultencoding('utf8')
nlp = spacy.load('fr')
client = MongoClient('localhost', 27007)
db = client.chatbot
collection = db.botmemory

# Fix Python 2.x.
try:
    input = raw_input
except NameError:
    pass


def chatterbot_initialize():
    """
    Initialize and train chatterbot function
    @output: bot (chatterbot entity)
    """
    bot = ChatBot(
        "Terminal",
        storage_adapter='chatterbot.storage.MongoDatabaseAdapter',
        logic_adapters=[
            {
                'import_path': 'chatterbot.logic.BestMatch',
                'response_selection_method':
                'chatterbot.response_selection.get_random_response'
            },
            {
                'import_path': 'chatterbot.logic.MathematicalEvaluation'
            }
        ],
        trainer='chatterbot.trainers.ChatterBotCorpusTrainer',
        database='chatbot',
        database_uri='mongodb://localhost:27007/'
    )
    bot.train("chatterbot.corpus.french")
    return bot


def chatterbot_get_response(bot, input_text):
    """ Answer from chatterbot corpus
    """
    return str(bot.get_response(input_text))


def rasa_initialize(model_directory):
    interpreter = Interpreter.load(model_directory,
                                   RasaNLUConfig("data/rasa/config_spacy.json"))
    return interpreter


def rasa_get_response(interpreter, input_text):
    confidence, intent = interpreter.parse(input_text)['intent'].values()
    return intent


def rasa_get_stories(intent):
    df = pd.read_csv('data/rasa/stories.csv', sep=';')
    print('intent', intent)
    if df['intent'].str.contains(intent).any():
        return ''.join(df[df['intent'] == intent]['answer'].values)
    else:
        return None


def read_slack(slack_client):
    """ Read messages from Real Time Messaging app

    args:
        slack_client (): Slack client to use API
    """
    return slack_client.rtm_read()


def connect_slack():
    """ Connect to slack app using token

    returns:
        Slack client object with token
    """
    return SlackClient(os.environ.get('SLACK_BOT_TOKEN'))


def clean_record():
    """ Initialize a clean record schema

    returns:
        record (dict): clean record schema
    """
    record = dict(input=dict(user="__default__", request=None),
                  output=dict(text=None), feedback=dict(reward=0))
    return record


def memorize(record, question, question_answer, user):
    """ Add record to memory db

    args:
        record (dict): Record with feedback
    """
    record["date"] = time.strftime('%d/%m/%Y-%H:%M')
    record["input"]["request"] = question
    record["input"]["user"] = user
    record["output"]["text"] = question_answer
    collection.insert(record)


def ask_reward(slack_client_token, question, question_answer, channel, BASH,
               record):
    """ Ask for reward

    returns:
        reward (int): 1 or -1 for good or bad reward
    TODO: Ask reward? multi conversations collition?
    """
    reward = None
    text = 'is my answer OK?'
    answer(slack_client_token, text, channel, BASH)
    # ask reward ???,
    record['feedback']['reward'] = reward
    memorize(record, question, question_answer, user)
    # input_user = input('is my answer OK ?')
    # if input_user == 'yes':
    #     reward = 1
    # elif input_user == 'no':
    #     reward = -1
    # return reward


def answer(slack_client_token, text, channel, BASH):
    """ Return answer either in slack or bash
    """
    if BASH:
        # record['output']['text'] = str(text)
        answer_bash(text)
        # reward = ask_reward()
        # record['feedback']['reward'] = reward
        # print(record)
        # if reward:
        #     memorize(record)
    else:
        answer_slack(slack_client_token, text, channel)


def answer_bash(text):
    """ Send the answer to the command line
    """
    print(text)


def answer_slack(slack_client, text, channel):
    """ Send the answer to slack
    """
    slack_client.api_call("chat.postMessage", channel=channel, text=text)


def parse_entities(command):
    """ Parse entities for a given phrase using spacy
    """
    doc = nlp(command)
    print("command", command, "doc", doc, "entities", doc.ents)
    if(doc.ents):
        response = 'Entities: '
        for ent in doc.ents:
            response += str(ent.text) + ", "
    else:
        response = "There are no entities in the phrase"
    print(response)
    return response


def available_velibs(place):
    """ Find the number of available velibs for a given station

    args:
        place (str): Place to obtain information about velib
    returns:
        resp (str): Information about the stations
    """
    response = requests.get("http://opendata.paris.fr/api/records/1.0/download/?dataset=stations-velib-disponibilites-en-temps-reel&facet=banking&facet=bonus&facet=status&facet=contract_name&rows=-1")
    txt = response.text
    f = open('data/velib.csv', 'w+')
    f.write(txt)
    velibs = pd.read_csv('data/velib.csv', sep=";")
    velibs = velibs[velibs.status == 'OPEN']
    subdf = velibs[velibs['name'].str.contains(place.upper())]
    resp = ''
    for num, row in subdf.iterrows():
        if row['status'] == 'OPEN':
            status_stand = ':white_check_mark: '
        elif row['status'] == 'CLOSED':
            status_stand = ':x: '

        resp = resp + status_stand + str(row['address']) + ', there are ' +\
            str(row['available_bikes']) + ' :bike: and ' +\
            str(row['available_bike_stands']) + ' :parking:\n'
    return resp


def compute_weather(place):
    """ Compute the weather (temperature and status) using a given location

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


def get_bot_id():
    """ Get bot id

    returns:
        botid (str): id number of the bot
    """
    # BOT_ID = os.environ.get("BOT_ID")
    # AT_BOT = "<@" + BOT_ID + ">"
    return os.environ.get("BOT_ID")


def list_channels(slack_client):
    print(json.dumps(slack_client.api_call("channels.list"), indent=2,
                     sort_keys=True))
