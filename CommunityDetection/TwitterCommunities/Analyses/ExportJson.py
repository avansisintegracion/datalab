# coding: UTF-8
#export en json pour la visualisation Vis.js.
#La fonction prend en entree un graphe Neo4j et le resultats d'une detection de communautes sous la fomrme d'un dictionnaire (communautes,liste de noeuds).
#La fonction cree un fichier json à l'emplavement precise (path) et retourne ce fichier.
import random as rand
import sys
reload(sys)
sys.setdefaultencoding('UTF8')

graph = Graph("http://192.168.1.75:7474/db/data/")

def exportJson(graph,Detec, path):
    comNodes={}
    Coo={}
    for x in Detec:
        Coo[x]=(rand.randint(-1000,1000),rand.randint(-1000,1000))
    for num,k in enumerate(graphe.vs):
        comNodes[k.index]=Detec[num]
    import json
    jsonFile={}
    jsonFile["nodes"]=[]
    jsonFile["edges"]=[]
    for link in graphe.es:
        jsonFile["edges"].append({"from": link.source,"to" : link.target})
    for node in graphe.vs:
        query = "MATCH (n{code:"+"'"+str(node["name"])+"'"+"}) RETURN n"
        res=graph.cypher.execute(query)
        if res[0][0]["name"] is None:
            jsonFile["nodes"].append({"id":node.index, "label":str(res[1][0]["name"]),"title" :"followers: "+str(res[1][0]["nb_followers"])+"<br>"+str(res[1][0]["description"]),"value":(PR['resPR'][node.index])*100000,"group" :comNodes[node.index],"x":Coo[comNodes[node.index]][0]+rand.randint(0,400),"y":Coo[comNodes[node.index]][1]+rand.randint(0,400)})
        else:
            jsonFile["nodes"].append({"id":node.index, "label":str(res[0][0]["name"]),"title" :"followers: "+str(res[0][0]["nb_followers"])+"<br>"+str(res[0][0]["description"]),"value":(PR['resPR'][node.index])*100000,"group" :comNodes[node.index],"x":Coo[comNodes[node.index]][0]+rand.randint(0,400),"y":Coo[comNodes[node.index]][1]+rand.randint(0,400)})
    with open(path, 'w') as f:
        json.dump(jsonFile, f, indent=4)
    f.close()

#export en json pour la visualisation Vis.js en supprimant les communautes de taille inferieure au parametre (taille).

def exportJson(graph,Detec, path,taille):


    comNodes={}
    Coo={}
    for x in IM['membership']:
        Coo[x]=(rand.randint(-1000,1000),rand.randint(-1000,1000))
    for num,k in enumerate(graphe.vs):
        comNodes[k.index]=IM['membership'][num]
    small=[]
    for item in comNodes.itervalues():
        if comNodes.values().count(item)<taille:
            small.append(item)
    import json
    jsonFile={}
    jsonFile["nodes"]=[]
    jsonFile["edges"]=[]
    for link in graphe.es:
        jsonFile["edges"].append({"from": link.source,"to" : link.target})
    for node in graphe.vs:
        if comNodes[node.index] not in small:
            query = "MATCH (n{code:"+"'"+str(node["name"])+"'"+"}) RETURN n"
            res=graph.cypher.execute(query)
            if res[0][0]["name"] is None:
                jsonFile["nodes"].append({"id":node.index, "label":str(res[1][0]["name"]),"title" :"followers: "+str(res[1][0]["nb_followers"])+"<br>"+str(res[1][0]["description"]),"value":(PR['resPR'][node.index])*100000,"group" :comNodes[node.index],"x":Coo[comNodes[node.index]][0]+rand.randint(0,400),"y":Coo[comNodes[node.index]][1]+rand.randint(0,400)})
            else:
                jsonFile["nodes"].append({"id":node.index, "label":str(res[0][0]["name"]),"title" :"followers: "+str(res[0][0]["nb_followers"])+"<br>"+str(res[0][0]["description"]),"value":(PR['resPR'][node.index])*100000,"group" :comNodes[node.index],"x":Coo[comNodes[node.index]][0]+rand.randint(0,400),"y":Coo[comNodes[node.index]][1]+rand.randint(0,400)})
    with open(path, 'w') as f:
        json.dump(jsonFile, f, indent=4)
    f.close()
