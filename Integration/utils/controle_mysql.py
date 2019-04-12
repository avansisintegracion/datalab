# coding: utf-8

import mysql.connector
from time import time
import sys

class database_sql:
    def __init__(self, host, user, password, database, log_key=None):
        self.connexion = None
        self.conn_cursor = None
        self.dataCo = {'host' : host, 'user' : user, 'password' : password, 'database' : database, 'raise_on_warnings' : True}
        self.table_name = database
        self.log = log_key
        self.query_insert = 'INSERT INTO random VALUES (NULL, %s, %s, %s, %s, %s)'
        self.list_result = []

    def launch_sql_database_connexion(self):
        connexion_check = None
        try:
            self.connexion = mysql.connector.connect(**self.dataCo)
        except(mysql.connector.Error, mysql.connector.Warning) as e:
            if self.log is not None:
                self.log.error(e) #ajout identifiant
            sys.exit(-1)

        connexion_check = self.connexion.is_connected()
        if connexion_check != True:
            if self.log is not None:
                self.log.error('Connexion a la base de donnees non actif')
            sys.exit(-1)

        self.conn_cursor = self.connexion.cursor()

    def close_sql_database_connexion(self):
        try:
            self.conn_cursor.close()
            self.connexion.close()
        except(mysql.connector.Error, mysql.connector.Warning) as e:
            if self.log is not None:
                self.log.error(e)
            sys.exit(-1)

    def drop_database(self):
        query_delete = 'DROP DATABASE IF EXISTS demo_ind'
        self.conn_cursor.execute(query_delete, ())
        self.connexion.commit()

    def drop_table(self):
        query_delete = 'TRUNCATE TABLE random'
        self.conn_cursor.execute(query_delete, ())
        self.connexion.commit()


    def insert_data_to_table(self, data_to_insert):
        self.conn_cursor.execute(self.query_insert, (data_to_insert[0], data_to_insert[0], 
            data_to_insert[1],data_to_insert[-1],data_to_insert[-1],))
        self.connexion.commit()

    def calcul_all_select(self):
        start = time()
        select_string = 'SELECT COUNT(*) FROM random WHERE first_letter = \'P\''
        self.conn_cursor.execute(select_string, ())
        end = time()
        for index in self.conn_cursor:
            pass
        self.list_result.append(end - start)

        start = time()
        select_string = 'SELECT COUNT(*) FROM random WHERE first_letter_index = \'P\''
        self.conn_cursor.execute(select_string, ())
        end = time()
        for index in self.conn_cursor:
            pass
        self.list_result.append(end - start)

        start = time()
        select_string = 'SELECT COUNT(*) FROM random WHERE value > 3700 AND value < 6800'
        self.conn_cursor.execute(select_string, ())
        end = time()
        for index in self.conn_cursor:
            pass
        self.list_result.append(end - start)

        start = time()
        select_string = 'SELECT COUNT(*) FROM random WHERE value_index > 3700 AND value_index < 6800'
        self.conn_cursor.execute(select_string, ())
        end = time()
        for index in self.conn_cursor:
            pass
        self.list_result.append(end - start)

        print(self.list_result)










