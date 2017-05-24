# coding: utf-8
import os
import spacy
import pyowm
import datetime
from pymongo import MongoClient
from slackclient import SlackClient

# starterbot's ID as an environment variable
BOT_ID = os.environ.get("BOT_ID")

# constants
AT_BOT = "<@" + BOT_ID + ">"

# instantiate Slack & Twilio clients
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))

nlp = spacy.load('en')
owm = pyowm.OWM('7491e475f58f9fab017349742ee03997')
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

def handle_command(command, channel):
    """
        Receives commands directed at the bot and determines if they
        are valid commands. If so, then acts on the commands. If not,
        returns back what it needs for clarification.
    """
    doc = nlp(command)
    print(doc, command, doc.ents)
    if(doc.ents):
        response = 'Entities: '
        for ent in doc.ents:
            response += str(ent.text) + ", "
    else:
        response = "Hello"
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)

def parse_slack_output(slack_rtm_output):
    """
        The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
    """
    output_list = slack_rtm_output
    channel = "#testbot"
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, whitespace removed
                return output['text'].split(AT_BOT)[1].strip(), \
                       output['channel']
            elif output['type'] == u'hello':
                text = "I'm back"
            elif output['type'] == u'member_joined_channel':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                text = "Welcome :simple_smile:" + talking_user
                answer_on_slack(text, channel)
            elif output['type'] == u'member_left_channel':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                text = "I am sad because " + talking_user + " left :weary:"
            elif output['type'] == u'goodbye':
                text = "Hasta la vista"
            elif output['type'] == u'message' and output['text'][0]==u'!':
                if output['type'] == u'message' and output['text'].split()[0] ==u'!weather':
                    place = output['text'].split()[1]
                    text = compute_weather(place)
                else:
                    talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                    text = "i know you are talking to me" + talking_user
            elif output['type'] == u'user_typing':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                text = "I know you are writting " + talking_user
            answer_on_slack(text, channel)
    return None, None


def compute_weather(place):
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


def answer(text):
    print(text)
    return None


def ask_reward():
    reward = None
    input_user = raw_input('is my answer OK ?')
    if input_user == 'yes':
        reward = 1
    elif input_user == 'no':
        reward = -1
    return reward


def answer_on_slack(text, channel):
    slack_client.api_call("chat.postMessage", channel=channel, text=text)
    return None


if __name__ == "__main__":
    while True:
        input_user = raw_input('Hello, I am listening... \n')
        record['input']['request'] = input_user
        bot_output = compute_weather(input_user)
        record['output']['text'] = bot_output
        answer(bot_output)
        reward = ask_reward()
        record['feedback']['reward'] = reward
        if reward:
            memorize(record)
