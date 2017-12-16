# Chatbot connected to Slack 

## Usage

* To test in local it _BASH_ flag in `startbot.py` should be _True_

* Initializate with  `python startbot.py`

* Initialize mongo deamon to save rewards:

`mongod --dbpath data --port 27007`

# Changelog

* [x] Say hello when arriving to channel 
* [x] Say hello when somebody arrives/leaves
* [x] Detection when somebody is writing
* [x] Entity recognition in a phrase.
* [x] Weather description for one city
* [x] Check available velibs around a given address
* [x] Handles multiple channels output
* [x] Ask reward in _BASH_ and _SLACK_ mode
* [ ] Save reward ? collision multi conversational inputs 


## Biblio

* [Get connected to slack](https://www.fullstackpython.com/blog/build-first-slack-bot-python.html)

* [Spacy entity recognition](https://spacy.io/docs/usage/entity-recognition)

* [Limbo functions for slack bot](https://github.com/llimllib/limbo)

* [Conversational bot module](https://github.com/gunthercox/ChatterBot)

* [Slack api methods](https://api.slack.com/methods)
