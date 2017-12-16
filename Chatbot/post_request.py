"""
Use slack web api with post requests
See https://api.slack.com/methods#users
"""
import requests
import json
import os
TOKEN = os.environ.get('SLACK_BOT_TOKEN')

if __name__ == "__main__":
    headers = {'Authorization': 'Bearer ' + TOKEN, 'Content-type': 'application/x-www-form-urlencoded'}  #'application/json'}
    data = {'token' : TOKEN, 'user': 'U57R0U3FV'}
    res = requests.post('https://slack.com/api/users.info', data=data, headers=headers)
    print(json.dumps(res.json(), indent=2))
