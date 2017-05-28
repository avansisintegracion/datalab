import os
from slackclient import SlackClient
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
import pyowm
owm = pyowm.OWM('7491e475f58f9fab017349742ee03997')  
import spacy
nlp = spacy.load('en')

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


def answer_bash(text):
    print(text)


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

def answer(text, channel, BASH, record):
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
    Parse entities
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
