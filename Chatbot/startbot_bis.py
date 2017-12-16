# coding: utf-8
import os
import time
from bot_functions import *

BASH = False

# Fix Python 2.x.
try: 
    input = raw_input
except NameError: 
    pass

def parse_slack_output(slack_client_token, output):
    """ Parse read output to extract message information
    Args: 
        slack_token (): Used to call slack api
        output (dict): Contains raw message
    Returns:
        dict: Parsed user, message, channel
    """
    user_id = output['user'] if 'user' in output else None
    if user_id:
        user_info = slack_client_token.api_call("users.info", user=user_id)['user']['profile']
    else:
        user_info = None
    user_name = user_info['first_name'] if (user_info and 'first_name' in user_info) else None
    message_type = output['type'] if 'type' in output else None
    message_content = output['text'] if 'text' in output else None
    channel = output['channel'] if 'channel' in output else None
    return dict(user_id=user_id, user_name=user_name,
                message_type=message_type, message_content=message_content,
                channel=channel)

def react_message(slack_client_token, message_info, record):
    """ React to a message 
    Args:
        slack_token (): Used to call slack api
        message_info (dict): Parsed user, message and channel
        record (dict): Feed back to save in database
    """
    ## Greetings
    if message_info['message_type'] == u'hello':
        text = "I'm back"
        #answer(slack_client_token, text,message_info['channel'], BASH)
    elif message_info['message_type'] == u'member_joined_channel':
        text = "Welcome :simple_smile:" + message_info['user_name']
        answer(slack_client_token, text,message_info['channel'], BASH)
    elif message_info['message_type'] == u'member_left_channel':
        text = "I am sad because " + message_info['user_name'] + " left :weary:"
        answer(slack_client_token, text,message_info['channel'], BASH)

    ## !Directed functions
    elif message_info['message_type'] == u'message' and message_info['message_content'][0]==u'!':
        if message_info['message_content'].split()[0] ==u'!weather':
            place = ' '.join(message_info['message_content'].split()[1:])
            try: 
                text = compute_weather(place)
            except:
                text = "I don't know where is " + place
            answer(slack_client_token, text,message_info['channel'], BASH)
            memorize(record, message_info['message_content'], text)
        elif message_info['message_content'].split()[0] ==u'!entities':
            text = parse_entities(unicode(' '.join(message_info['message_content'].split()[1:])))
            answer(slack_client_token, text,message_info['channel'], BASH)
            memorize(record, message_info['message_content'], text)
        elif message_info['message_content'].split()[0] ==u'!velib':
            place = ' '.join(message_info['message_content'].split()[1:])
            text = available_velibs(place)
            answer(slack_client_token, text,message_info['channel'], BASH)
            memorize(record, message_info['message_content'], text)
        else:
            text = "i know you are talking to me " + message_info['user_name']
            answer(slack_client_token, text, message_info['channel'], BASH)
            ask_reward(slack_client_token, message_info['message_content'], 'i know your talking to me', message_info['channel'], BASH, record)
    elif message_info['message_type'] == u'user_typing':
        pass
    ## General conversation
    elif message_info['message_type'] == u'message' and message_info['user_id']:
        text = chatterbot_get_response(bot, message_info['message_content']) 
        print('bot ans',text)
        answer(slack_client_token, text,message_info['channel'], BASH)

if __name__ == "__main__":
    bot = initialize_chatterbot()
    BOT_ID = get_bot_id()
    if BASH == True:
        print('Hello I am listening')
        while True:
            input_user = input()
            record = clean_record()
            record['input']['request'] = input_user
            #input_chatbot = {'type' : u'message', 'text': input_user, 'user': u'bot'}
            input_chatbot = dict(user_id=u'bot', user_name=u'bot',
                        message_type='message', message_content=input_user,
                        channel=u'empty')
            react_message(None, input_chatbot, record)
    else:
        READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
        slack_client_token = connect_slack()
        if slack_client_token.rtm_connect():
            print("StarterBot connected and running!")
            while True:
                #try:
                read_output = read_slack(slack_client_token)
                if read_output:
                    for message in read_output:
                        print('before parsing', message)
                        record = clean_record()
                        message_info = parse_slack_output(slack_client_token, message)
                        print('inside', message_info)  
                        react_message(slack_client_token, message_info, record)
                time.sleep(READ_WEBSOCKET_DELAY)
                #except KeyboardInterrupt:
                #    print 'Interrupted'
                #    os._exit(0)
                #except:
                #    #print('error in parse slack ' + time.strftime('%m/%d-%H-%M'))
                #    pass
        else:
            print("Connection failed. Invalid Slack token or bot ID?")
