# coding: utf-8

from py2neo import Graph
from time import time
import sys

class database_neo:
    def __init__(self, host, port, user, password, log_key=None):
        self.graph = None
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.log = log_key
        self.dict_result = {}
        self.list_result = []

    def launch_neo_database_connexion(self):
        host_value = self.host + ':' + self.port
        self.graph = Graph(host=host_value, auth=(self.user, self.password))

    def create_node(self, data_to_insert):
        node_label = 'new_alpha'
        identifiant = 'na_1'
        self.dict_result['first_letter'] = data_to_insert[0]
        self.dict_result['first_letter_index'] = data_to_insert[0]
        self.dict_result['string'] = data_to_insert[1]
        self.dict_result['value'] = data_to_insert[-1]
        self.dict_result['value_index'] = data_to_insert[-1]

        parameter_dict = {'params': self.dict_result}
        query = 'CREATE ({0}:{1} {2}) RETURN {0}'.format(identifiant ,node_label, '{params}')
        self.graph.run(query, parameters=parameter_dict)

    def create_all_index(self):
        node_label = 'new_alpha'
        attribut = 'first_letter_index'
        query_ind = 'CREATE INDEX ON :{0}({1})'.format(node_label, attribut)
        self.graph.run(query_ind)

        attribut = 'value_index'
        query_ind = 'CREATE INDEX ON :{0}({1})'.format(node_label, attribut)
        self.graph.run(query_ind)

    def calcul_all_search(self):
        node_label = 'new_alpha'
        identifiant = 'na_1'
        attribut = 'first_letter'
        query_match = "MATCH ({0}:{1}) WHERE {0}.{2} = {3} RETURN count({0}) as count".format(identifiant, node_label, attribut, "'P'")
        start = time()
        self.graph.run(query_match)
        end = time()
        self.list_result.append(end - start)

        attribut = 'first_letter_index'
        query_match = "MATCH ({0}:{1}) WHERE {0}.{2} = {3} RETURN count({0}) as count".format(identifiant, node_label, attribut, "'P'")
        start = time()
        self.graph.run(query_match)
        end = time()
        self.list_result.append(end - start)
        
        attribut = 'value'
        query_match = 'MATCH ({0}:{1}) WHERE {0}.{2} > 3700 AND {0}.{2} < 6800 RETURN count({0}) as count'.format(identifiant, node_label, attribut)
        start = time()
        self.graph.run(query_match)
        end = time()
        self.list_result.append(end - start)

        attribut = 'value_index'
        query_match = 'MATCH ({0}:{1}) WHERE {0}.{2} > 3700 AND {0}.{2} < 6800 RETURN count({0}) as count'.format(identifiant, node_label, attribut)
        start = time()
        self.graph.run(query_match)
        end = time()
        self.list_result.append(end - start)

        print(self.list_result)

    def delete_all_data(self):
        query_delete = 'MATCH (n) DETACH DELETE n'
        self.graph.run(query_delete)









