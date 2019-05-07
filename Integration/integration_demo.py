# coding: utf-8

import sys
import logging
import string

from utils.controle_mysql import *
from utils.controle_mongo import *
from utils.controle_neo import *
from utils.controle_log import *
from random import randint
from random import SystemRandom

def generate_random_value():
    list_value = []
    string_value = ''.join(SystemRandom().choice(string.ascii_uppercase) for _ in range(10))
    letter_value = string_value[0]
    integer_value = randint(0,10000)

    list_value.append(letter_value)
    list_value.append(string_value)
    list_value.append(integer_value)

    return(list_value)

if __name__ == "__main__":
    log_controle = set_logger()
    log = log_controle.initialize_logger()
    
    sql_controle = database_sql('localhost', 'mkone', 'azerty', 'demo_ind', log)
    sql_controle.launch_sql_database_connexion()
    sql_controle.drop_table()
    
    mongo_controle = database_mongo('localhost', '27017', log)
    mongo_controle.launch_mongo_database_connexion()
    mongo_controle.drop_database()
    mongo_controle.create_mongo_collection()

    neo_controle = database_neo('localhost', '7687', 'neo4j', 'proviprovi', log)
    neo_controle.launch_neo_database_connexion()
    neo_controle.delete_all_data()
    neo_controle.create_all_index()

    for i in range(500000):
        verif = generate_random_value();
        sql_controle.insert_data_to_table(verif)
        mongo_controle.insert_data_to_collection(verif)
        neo_controle.create_node(verif)
    
    print("letter without index -> letter with index -> value without index -> value with index")
    sql_controle.calcul_all_select()
    mongo_controle.calcul_all_select()
    neo_controle.calcul_all_search()

    sql_controle.close_sql_database_connexion()
    mongo_controle.close_mongo_database_connexion()

    sys.exit(0)
