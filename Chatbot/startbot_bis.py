# coding: utf-8
import os
import spacy
import pyowm
import time
from slackclient import SlackClient
from bot_functions import *
from chatterbot import ChatBot

bot = ChatBot(
    "Terminal",
    logic_adapters=[
        "chatterbot.logic.BestMatch",
        "chatterbot.logic.MathematicalEvaluation"
        #"chatterbot.logic.TimeLogicAdapter"
    ],
    trainer='chatterbot.trainers.ChatterBotCorpusTrainer',
)

# Train based on the english corpus
bot.train("chatterbot.corpus.english")
#bot.train("chatterbot.corpus.english.greetings")


# Slack
BASH = False
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
#BOT_ID = os.environ.get("BOT_ID") # starterbot's ID as an environment variable
#AT_BOT = "<@" + BOT_ID + ">" # constants
# Fix Python 2.x.
try: 
    input = raw_input
except NameError: 
    pass

def parse_slack_output(output_list, record):
    """
        The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
    """
    channel = "#testbot"
    if output_list and len(output_list) > 0:
        for output in output_list:
            try:
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                msj = output['text']
                print(talking_user, msj)
            except:
                pass
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
                    place = ' '.join(output['text'].split()[1:])
                    try: 
                        text = compute_weather(place)
                    except:
                        text = "I don't know where is " + place
                    answer(text, channel, BASH, record)
                elif output['type'] == u'message' and output['text'].split()[0] ==u'!entities':
                    text = parse_entities(unicode(' '.join(output['text'].split()[1:])))
                    answer(text, channel, BASH, record)
                elif output['type'] == u'message' and output['text'].split()[0] ==u'!velib':
                    place = ' '.join(output['text'].split()[1:])
                    text = available_velibs(place)
                    answer(text, channel, BASH, record)
                else:
                    talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                    text = "i know you are talking to me" + talking_user
                    answer(text, channel, BASH, record)
            elif output['type'] == u'user_typing':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                print(talking_user, " is typing")
            elif output['type'] == u'message':
                print('messaj', str(output['text']), 'user', output['user'])
                text = bot.get_response(output['text'])
                print('bot ans',str(text))
                answer(str(text), channel, BASH, record)

if __name__ == "__main__":
    if BASH == True:
        while True:
            input_user = input('Hello, I am listening... \n')
            record = clean_record()
            record['input']['request'] = input_user
            input_chatbot = [{'type' : u'message', 'text': input_user, 'user': u'bot'}]
            parse_slack_output(input_chatbot, record)
    else:
        READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
        if slack_client.rtm_connect():
            print("StarterBot connected and running!")
            while True:
                try:
                    record = clean_record()
                    parse_slack_output(slack_client.rtm_read(), record)
                    time.sleep(READ_WEBSOCKET_DELAY)
                except:
                    pass
                    #print('error in parse slack ' + time.strftime('%m/%d-%H-%M'))
        else:
            print("Connection failed. Invalid Slack token or bot ID?")
