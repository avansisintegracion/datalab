from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

training_data = load_data('data/rasa/testData.json')
trainer = Trainer(RasaNLUConfig("data/rasa/config_spacy.json"))
trainer.train(training_data)
model_directory = trainer.persist('data/rasa/models/')
