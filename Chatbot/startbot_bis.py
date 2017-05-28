# coding: utf-8
import os
import spacy
import pyowm
import time
from slackclient import SlackClient
from pymongo import MongoClient
from bot_functions import *

# Slack
BASH = False
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
BOT_ID = os.environ.get("BOT_ID") # starterbot's ID as an environment variable
AT_BOT = "<@" + BOT_ID + ">" # constants
# Spacy
nlp = spacy.load('en')
# Python Open weather maps
owm = pyowm.OWM('7491e475f58f9fab017349742ee03997')
# Mongo 
client = MongoClient('localhost', 27017)
db = client.chatbot
collection = db.botmemory
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

def parse_slack_output(output_list):
    """
        The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
    """
    channel = "#testbot"
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output['type'] == u'hello':
                text = "I'm back"
                answer(text, channel, BASH, record)
            elif output['type'] == u'member_joined_channel':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                text = "Welcome :simple_smile:" + talking_user
                answer(text, channel, BASH, record)
            elif output['type'] == u'member_left_channel':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                text = "I am sad because " + talking_user + " left :weary:"
                answer(text, channel, BASH, record)
            elif output['type'] == u'goodbye':
                text = "Hasta la vista"
                answer(text, channel, BASH, record)
            elif output['type'] == u'message' and output['text'][0]==u'!':
                if output['type'] == u'message' and output['text'].split()[0] ==u'!weather':
                    place = output['text'].split()[1]
                    text = compute_weather(place)
                    answer(text, channel, BASH, record)
                elif output['type'] == u'message' and output['text'].split()[0] ==u'!entities':
                    text = parse_entities(unicode(' '.join(output['text'].split()[1:])))
                    answer(text, channel, BASH, record)
                else:
                    talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                    text = "i know you are talking to me" + talking_user
                    answer(text, channel, BASH, record)
            elif output['type'] == u'user_typing':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                text = "I know you are writting " + talking_user
                answer(text, channel, BASH, record)



if __name__ == "__main__":
    if BASH == True:
        while True:
            input_user = raw_input('Hello, I am listening... \n')
            record['input']['request'] = input_user
            input_chatbot = [{'type' : u'message', 'text': input_user}]
            parse_slack_output(input_chatbot)
    else:
        READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
        if slack_client.rtm_connect():
            print("StarterBot connected and running!")
            while True:
                parse_slack_output(slack_client.rtm_read())
                time.sleep(READ_WEBSOCKET_DELAY)
        else:
            print("Connection failed. Invalid Slack token or bot ID?")
