# -*- coding: UTF-8 -*-`

import igraph
from py2neo import Graph, authenticate

#La fonction prend en paramètres le membership issu de l'algorithme de detection de communautés et le graphe Twitter.
#Elle renvoit un dictionnaire contenant lui-même deux dictionnaires. Le premier associe les communautés aux comptes
#Twitter qui leur appartiennent, et le second associe les communautés aux noms des noeuds qui leur appartiennent.
def NomsCommunautes(membership,graphe):
    communautes={}
    keys=[]
    for indx,node in enumerate(graphe.vs):
        if membership[indx] not in keys:
            communautes[membership[indx]]=[]
            communautes[membership[indx]].append(node['name'])
            keys.append(membership[indx])
        else:
            communautes[membership[indx]].append(node['name'])
    communautesAvecNoms={}
    graph = Graph("http://192.168.1.75:7474/db/data/")
    keys2=[]
    for key,obj in communautes.iteritems():
        for code in obj:
            query2 = "MATCH (n{code:"+"'"+str(code)+"'"+"}) RETURN n.name"
            res=graph.cypher.execute(query2)
            if key in keys2:
                communautesAvecNoms[key].append(res)
            else:
                communautesAvecNoms[key]=[]
                communautesAvecNoms[key].append(res)
                keys2.append(key)
    return {'comNoms':communautesAvecNoms,'comNode':communautes}

#Cette fonction sert à récupérer un pourcentage des comptes les plus influents dans les commaunutés détectées au sein
#du graphe Twitter. Elle prend en paramètres le résultat direct du PageRank, le dictionnaire communautés-noms des noeuds,
#le pourcentage de comptes les plus influents que l'on veut récupérer et la taille minimale des communautés dont on
#veut récupérer les comptes les plus influents. La fonction retourne un dictionnaire qui associe chaque commuanuté à
#la liste de ses comptes les plus influents (proportion paramétrée). Si le pourcentage donné est trop faible par rapport
#a la taille minimale donnée, les communautés ayant la taille minimale ne seront pas prises en compte.
def InfluenceCommunautes(graphe,graphetxt,PR,comNode,percentage,tailleMin):

    Influence={}

    nodePR={}
    for indx, value in enumerate(PR):

        nodePR[graphetxt.vs[indx]['name']]=value

    for key, value in comNode.iteritems():
         if len(value)>=tailleMin:

            arr=[]
            nb=int(len(value)*percentage)


            for i in value:

                tuplee=(i,nodePR[i])
                arr.append(tuplee)

            arr= sorted(arr, key=lambda tup: tup[1],reverse=True)
            if nb>1:
                arr=arr[0:nb-1]
            else:
                continue

            nodes=[]

            for tuplee in arr:
                query = "MATCH (n{code:"+"'"+str(tuplee[0])+"'"+"}) RETURN n.name"
                res=graphe.cypher.execute(query)
                if res[0][0] is not None:
                    nodes.append(res[0][0])
                else:
                    nodes.append(res[1][0])

            Influence[key]=nodes

    return Influence

#Cette fonction permet de récupérer un dictionnaire des hashtags les plus récurrents pour chaque communautés. On doit préciser
#le pourcentage de hashtags par communautés que l'on veut récupérer. On passe également en paramètre le dictionnaire (id_com,liste id_noeuds).
def InfluenceHashtags(graphe,comNode,maxim):
    import operator
    from operator import itemgetter

    Influence={}

    for key in comNode.iterkeys():

        liste=[]
        for value in comNode[key]:

            query="MATCH (n{code:"+"'"+str(value)+"'"+"})<-[r:`andra-from`]-(t:`andra-tweet`)-[r2:`andra-#about`]->(h:`andra-hashtag`) RETURN h.code"
            res=graphe.cypher.execute(query)

            for el in res:

                liste.append(el[0])
        Influence[key]=liste


    dico={}

    for com in Influence.iterkeys():
        keys=[]
        liste=[]
        for h in Influence[com]:
            if h not in keys:
                tuplee=(h,Influence[com].count(h))
                liste.append(tuplee)
                keys.append(h)
        dico[com]=liste

    for com in dico.iterkeys():

        dico[com]= sorted(dico[com], key=lambda tup: tup[1],reverse=True)
        N=len(dico[com])
        if N>maxim:
            dico[com]=dico[com][:maxim]

    return dico


#Cette fonction permet de récupérer un dictionnaire des tweets les plus retweetés pour chaque communautés. On doit préciser
#On passe en paramètre le dictionnaire (id_com,liste id_noeuds).
def InfluenceTweets(comNode):
    import operator
    from operator import itemgetter

    graph = Graph("http://192.168.1.75:7474/db/data/")
    Influence={}

    for key in comNode.iterkeys():

        liste=[]
        for value in comNode[key]:

            query="MATCH (n{code:"+"'"+str(value)+"'"+"})<-[r:`andra-from`]-(t:`andra-tweet`)<-[r1:`andra-retweet`]-(a:`andra-tweet`) RETURN t.text, count(r1)"
            res=graph.cypher.execute(query)


            for el in res:
                tuplee=(el[0],el[1])
                liste.append(tuplee)
        Influence[key]=liste

        for com in Influence.iterkeys():

            Influence[com]= sorted(Influence[com], key=lambda tup: tup[1],reverse=True)
            Influence[com]=Influence[com][:3]

    return Influence
