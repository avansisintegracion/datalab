# coding: utf-8

import pymongo
from time import time
from bson.objectid import ObjectId
import sys

class database_mongo:
    def __init__(self, host, port, log_key=None):
        self.client = None
        self.database = None
        self.collection = None
        self.host = host
        self.port = port
        self.dataCo = ''
        self.log = log_key
        self.dict_result = {}
        self.list_result = []

    def launch_mongo_database_connexion(self):
        self.dataCo = 'mongodb://' + self.host + ':' + self.port + '/'
        self.client = pymongo.MongoClient(self.dataCo)

    def create_mongo_collection(self):
        self.database = self.client["demo_ind"]
        self.collection = self.database["random"]
        self.collection.create_index("first_letter_index", name="ind_first_letter")
        self.collection.create_index("value_index", name="ind_value")

    def close_mongo_database_connexion(self):
        self.client.close()

    def drop_database(self):
        self.client.drop_database('demo_ind')

    def insert_data_to_collection(self, data_to_insert):
        self.dict_result['_id'] = ObjectId()
        self.dict_result['first_letter'] = data_to_insert[0]
        self.dict_result['first_letter_index'] = data_to_insert[0]
        self.dict_result['string'] = data_to_insert[1]
        self.dict_result['value'] = data_to_insert[-1]
        self.dict_result['value_index'] = data_to_insert[-1]
        self.collection.insert_one(self.dict_result)

    def calcul_all_select(self):
        query = {'first_letter' : 'P'}
        start = time()
        verif = self.collection.find(query).count()
        end = time()
        self.list_result.append(end - start)

        query = {'first_letter_index' : 'P'}
        start = time()
        verif = self.collection.find(query).count()
        end = time()
        self.list_result.append(end - start)

        query = { 'value' : { '$gt': 3700, '$lt': 6800 } }
        start = time()
        verif = self.collection.find(query).count()
        end = time()
        self.list_result.append(end - start)

        query = { 'value_index' : { '$gt': 3700, '$lt': 6800 } }
        start = time()
        verif = self.collection.find(query).count()
        end = time()
        self.list_result.append(end - start)

        print(self.list_result)
