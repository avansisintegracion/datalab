import os
import time
from slackclient import SlackClient
import spacy
from spacy.en import English
import ipdb
nlp = spacy.load('en')
import pyowm
owm = pyowm.OWM('7491e475f58f9fab017349742ee03997')  


# starterbot's ID as an environment variable
BOT_ID = os.environ.get("BOT_ID")

# constants
AT_BOT = "<@" + BOT_ID + ">"
#EXAMPLE_COMMAND = "do"

# instantiate Slack & Twilio clients
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))


def handle_command(command, channel):
    """
        Receives commands directed at the bot and determines if they
        are valid commands. If so, then acts on the commands. If not,
        returns back what it needs for clarification.
    """
    doc = nlp(command)
    print(doc,command, doc.ents)
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
    #print(output_list)
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, whitespace removed
                return output['text'].split(AT_BOT)[1].strip(), \
                       output['channel']
            elif output['type'] == u'hello':
                slack_client.api_call("chat.postMessage", channel="#testbot", text="I'm back")
            elif output['type'] == u'member_joined_channel':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                slack_client.api_call("chat.postMessage", channel="#testbot", text="Welcome :simple_smile:" + talking_user)
            elif output['type'] == u'member_left_channel':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                slack_client.api_call("chat.postMessage", channel="#testbot", text="I am sad because " + talking_user + " left :weary:" )
            elif output['type'] == u'goodbye':
                slack_client.api_call("chat.postMessage", channel="#testbot", text="Hasta la vista")
            elif output['type'] == u'message' and output['text'][0]==u'!':
                if output['type'] == u'message' and output['text'].split()[0] ==u'!weather':
                    observation = owm.weather_at_place(output['text'].split()[1])  
                    w = observation.get_weather()  
                    temperature = w.get_temperature('celsius')['temp']  
                    slack_client.api_call("chat.postMessage", channel="#testbot", text="It is " + str(w.get_detailed_status()) + " and it is " + str(temperature) + "C " + "in " + observation.get_location().get_name())
                else:
                    talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']

                    slack_client.api_call("chat.postMessage", channel="#testbot", text="i know you are talking to me" + talking_user)
            elif output['type'] == u'user_typing':
                talking_user = slack_client.api_call("users.info", user=output['user'])['user']['profile']['first_name']
                slack_client.api_call("chat.postMessage", channel="#testbot", text="I know you are writting " + talking_user)
    return None, None



if __name__ == "__main__":
    READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
    if slack_client.rtm_connect():
        print("StarterBot connected and running!")
        #ipdb.set_trace()
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                handle_command(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")
