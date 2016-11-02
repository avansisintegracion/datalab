#Fonction qui renvoit une visualisation de la partition issue de la détection de communautés sur le graphe.
def VisualisationIG(vertexClustering):
    return ig.plot(vertexClustering,vertex_size=6,bbox = (500, 500),edge_arrow_size=0.5)

def VisualisationNX(commClus,graphe):
    import random

    communautes={}
    keys=[]
    for indx,nodes in commClus.iteritems():
        col=str('#') +str('%06X' % random.randint(0, 0xFFFFFF))
        for node in nodes:
            if node not in keys:

                communautes[str(node)]=col
                keys.append(str(node))
            else:
                communautes[str(node)]="black"



        for node in graphe.vs:
            if str(node["name"]) not in communautes.keys():

                communautes[node["name"]]="white"


    graphe.vs["color"]=[communautes[node] for node in graphe.vs["name"]]

    return ig.plot(graphe,vertex_size=6,bbox = (500, 500),edge_arrow_size=0.5)


def VisualisationSNAP(commClus,graphe):
    import random


    communautes={}
    keys=[]
    for indx,nodes in commClus.iteritems():
        col=str('#') +str('%06X' % random.randint(0, 0xFFFFFF))
        for node in nodes:
            if graphe.vs[int(node)]["name"] not in keys:

                communautes[graphe.vs[int(node)]["name"]]=col
                keys.append(graphe.vs[int(node)]["name"])
            else:
                communautes[graphe.vs[int(node)]["name"]]="black"



        for node in graphe.vs:
            if str(node["name"]) not in communautes.keys():

                communautes[node["name"]]="white"


    graphe.vs["color"]=[communautes[node] for node in graphe.vs["name"]]

    return ig.plot(graphe,vertex_size=6,bbox = (500, 500),edge_arrow_size=0)
