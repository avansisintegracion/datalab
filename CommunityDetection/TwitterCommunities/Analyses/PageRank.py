# coding: utf-8

# Fonctions de classement de l'influence des comptes utilisateurs et tweets selon l'algorithme PageRank d' Igraph,
#et classement des hashtags (par nombre d'occurences).



import operator
import heapq
import numpy as np
from py2neo import Graph, authenticate
import igraph as ig

#Fonction de récupération d'un graphe en format csv à partir d'une base de données Neo4j, puis création d'un objet
#Igraph. La fonction prend en paramètre une requête Cypher qui récupère des relations uniques entre noeuds du même
#type afin de génerer un nouveau graphe à partir de ces données. On doit preciser l'adresse du graphe, le port et
#l'authentification à la base Neo4j.

def getIGraph(query,graph,port,ident,mdp):


    authenticate(port, ident, mdp)
    query = query
    with open('test_dump_subgraph.txt','w') as file:
        for record in graph.cypher.execute(query):
            node_from = record.nodeFrom.properties['code']
            node_to = record.nodeTo.properties['code']
            file.write(str(node_from) +"\t"+ str(node_to) +"\n")


    g = ig.Graph.Read_Ncol('test_dump_subgraph.txt')
    return g

#La fonction prend plusieurs paramètres : le graphe issu de la fonction précédente, Le type de noeud que l'on veut
#classer (utilisateurs, tweets ou hashtags) et la taille du classement que l'on souhaite. Pour les utilisateurs et
#les tweets, L'algorithme PageRank d'Igraph est appliqué et on récupère un classement (de bas en haut) ainsi que les
#scores pageRank associés (et la date pour les tweets). Pour les hashtags, on les classe par rapport à leur nombre
#d'occurences. Le fonction renvoie un dictionnaire dont les élements sont le classement issu du PageRank et le résultats
#direct du PageRank(liste de valeurs)

def Rank(g,nbClass):

    PR=g.pagerank(weights=None)

    arr=np.array([])
    for i in PR:
        arr=np.append(arr,i)
    arr=arr.argsort()[-nbClass:][::-1]
    codes=[]
    for indx in arr:
        codes.append(g.vs[indx]['name'])
    return {'classement':codes,'resPR':PR}

#Possibilités de requêtes :
#MATCH (n:`andra-tweet`)-[re:`andra-retweet`]->(p:`andra-tweet`) RETURN n as nodeFrom,p as nodeTo
#MATCH (n:`andra-user`)<-[r:`andra-from`]-(t1:`andra-tweet`)-[re:`andra-retweet`]->(t2:`andra-tweet`)-[rel:`andra-from`]->(p:`andra-user`) WHERE t2.language="'"fr"'" RETURN n as nodeFrom,p as nodeTo
#MATCH (n:`andra-tweet`)-[r:`andra-quote`]->(p:`andra-tweet`) RETURN n as nodeFrom,p as nodeTo
#MATCH (n:`andra-user`)<-[r:`andra-from`]-(t:`andra-tweet`)-[re:`andra-@mention`]->(p:`andra-user`) RETURN n as nodeFrom,p as nodeTo"
#MATCH (k)-[m:`DAY`]->(d)<-[r:`Date`]-(n:`andra-tweet`)-[q:`andra-quote`]->(p:`andra-tweet`) WHERE d.day=2 and k.month=6 RETURN n as nodeFrom,p as nodeTo
