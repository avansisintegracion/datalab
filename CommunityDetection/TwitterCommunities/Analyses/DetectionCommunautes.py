#coding: UTF-8
#Fonctions qui appliquent un algorithme de détection de communauté choisi au graphe donné en entrée.
# La fonction DetectionComIG permet d'appliquer des algorithmes de la librairie Igraph, les choix pour le paramètre "algo"
#sont : InfoMap et FastGreedy.La fonction renvoit un dictionnaire contenant un objet VertexClustering et une liste membership.

from py2neo import Graph, authenticate
import MiseEnFormeSortieFScore as mise

def DetectionComIG(graphe, algo):

    if algo=="InfoMap":
        ResClus=graphe.community_infomap()
        membership=ResClus.membership

    elif algo=="FastGreedy":
        graphe=graphe.as_undirected()
        for es in graphe.es:
            es["weight"]=1
        graphe=graphe.simplify(combine_edges=dict(weight="sum"))
        clus=graphe.community_fastgreedy()
        ResClus=clus.as_clustering()
        membership=ResClus.membership
    return {'ResClus':ResClus,'membership':membership}


#La fonction DetectionComNX applique l'agorithme kclique de Networkx au graphe donne en entree, on récupère en sortie un dictionnaire
#{communaute,liste de noeuds}.

def DetectionComNX(graphe,path):

    import networkx as nx

    graph=nx.read_edgelist(path+graphe)
    kcliques=nx.k_clique_communities(graph,3,nx.find_cliques(graph))
    c=list(kcliques)
    commClus={}
    for indx,val in enumerate(c):
        temp=[]
        for j in val:
            temp.append(int(j.replace("n","")))
        commClus[indx]=temp
    return commClus

# La fonction DetectionComSNAP permet d'appliquer des algorithmes de la librairie SNAP, les choix pour le paramètre "algo"
#sont : BigClam, InfoMap et CPM. Il faut préciser le chemin vers SNAP, le chemin vers le repertoire global, le nom du fichier contenant
#le graphe csv et le nom contenant l'output de l'algorithme (ex : res.txt).
#La fonction renvoit deux dictionnaires : le premier (communaute,liste de noeuds) et le second(communaute,liste
#de noms utilisateurs).

def DetectionComSNAP(graphe,algo,SnapPath,path,outputGraph,outputAlgo):

    import subprocess as sb
    import os
    text_file = open(path+outputGraph, "w")
    for es in graphe.es:

            text_file.write(str(es.tuple[0])+"\t"+str(es.tuple[1])+"\n")
    text_file.close()
    commClus1={}
    commClus2={}

    if algo=="BigClam":
        sb.call("./bigclam -o:"+outputAlgo.replace(".txt","")+" -i:"+path+outputGraph+" -c:-1 -mc:2 -xc:100", cwd=SnapPath+'examples/bigclam',shell=True)
        text_file = open(SnapPath+"examples/bigclam/"+outputAlgo.replace(".txt","")+"cmtyvv.txt", "r")

        for indx,line in enumerate(text_file):
            line = line.rstrip()
            fields = line.split("\t")
            noeuds=[]
            for col in fields:

                noeuds.append(graphe.vs[int(col)]['name'])
            commClus1[indx]=noeuds


    elif algo=="CPM":
        sb.call(SnapPath+"examples/cliques/./cliquesmain -o:"+outputAlgo.replace(".txt","")+" -i:"+path+outputGraph+" -k:2", cwd=SnapPath+"examples/cliques",shell=True)
        text_file = open(SnapPath+"examples/cliques/cpm-"+outputAlgo, "r")
        for indx,line in enumerate(text_file):
            if indx>1:
                line = line.rstrip()
                fields = line.split("\t")
                noeuds=[]
                for col in fields:
                    noeuds.append(graphe.vs[int(col)]['name'])
                commClus1[indx]=noeuds

    elif algo=="InfoMap":
        sb.call("./community -i:"+path+outputGraph+" -a:3", cwd=SnapPath+"/examples/community",shell=True)
        mise.MiseEnFormeSortie(SnapPath+"/examples/community/communities.txt",
        SnapPath+"/examples/community/"+outputAlgo)
        text_file = open(SnapPath+"/examples/community/"+outputAlgo, "r")
        for indx,line in enumerate(text_file):
            line = line.rstrip()
            fields = line.split("\t")
            noeuds=[]
            for col in fields:
                noeuds.append(graphe.vs[int(col)]['name'])
            commClus1[indx]=noeuds
        text_file.close()

    authenticate("192.168.1.75:7474", "neo4j", "pass4dbse")
    graph = Graph("http://192.168.1.75:7474/db/data/")
    for key,value in commClus1.iteritems():
                values=[]
                for val in value:
                    query="MATCH (n {code:"+"'"+str(val)+"'"+"}) return n.name"
                    res=graph.cypher.execute(query)
                    for c in res:
                        res=c
                    values.append(res)
                commClus2[key]=values


    text_file.close()
    return {'comCodes':commClus1,'comNoms':commClus2}
