# -*- coding: UTF-8 -*-`

import DetectionCommunautes
import PageRank
import PerformanceFScore
import InfluenceCommunautes
import igraph
from py2neo import Graph, authenticate

path="/Users/Charlotte/Documents/"
authenticate('192.168.1.75:7474', 'neo4j', 'pass4dbse')
grapheNeo = Graph("http://192.168.1.75:7474/db/data/")
query = 'MATCH (n:`andra-user`)<-[r:`andra-from`]-(t1:`andra-tweet`)-[re:`andra-retweet`]->(t2:`andra-tweet`)-[rel:`andra-from`]->(p:`andra-user`) WHERE t2.language="'"fr"'" RETURN n as nodeFrom,p as nodeTo'
with open(path+'test_dump_subgraph.txt','w') as file:
    for record in grapheNeo.cypher.execute(query):
        node_from = record.nodeFrom.properties['code']
        node_to = record.nodeTo.properties['code']
        file.write(str(node_from) +"\t"+ str(node_to) +"\n")

gTwitter = igraph.Graph.Read_Ncol(path+'test_dump_subgraph.txt')
GraphFile='com-dblp.ungraph.txt'
GTFile='com-dblp.all.cmty.txt'

def test_Detec():


     resIGFG=DetectionCommunautes.DetectionComIG(gTwitter,"FastGreedy")
     resIGIM=DetectionCommunautes.DetectionComIG(gTwitter,"InfoMap")

     assert type(resIGFG)==dict
     assert len(resIGFG)==2
     assert type(resIGFG['ResClus'])==igraph.VertexClustering
     assert type(resIGFG['membership'])==list
     assert len(resIGFG['membership']) !=0

     assert type(resIGIM)==dict
     assert len(resIGIM)==2
     assert type(resIGIM['ResClus'])==igraph.VertexClustering
     assert type(resIGIM['membership'])==list
     assert len(resIGIM['membership']) !=0

     resNX=DetectionCommunautes.DetectionComNX(GraphFile,path)

     assert type(resNX)==dict
     assert len(resNX)!=0

     resSNBC=DetectionCommunautes.DetectionComSNAP(gTwitter,"BigClam",path+'snap/',path,'fichierGraphe.txt','resAlgo.txt')
     resSNCPM=DetectionCommunautes.DetectionComSNAP(gTwitter,"CPM",path+'snap/',path,'fichierGraphe.txt','resAlgo.txt')
     resSNIM=DetectionCommunautes.DetectionComSNAP(gTwitter,"InfoMap",path+'snap/',path,'fichierGraphe.txt','resAlgo.txt')

     assert type(resSNBC)==dict
     assert len(resSNBC)==2
     assert type(resSNBC['comCodes'])==dict
     assert type(resSNBC['comNoms'])==dict
     assert len(resSNBC['comCodes'])!=0
     assert len(resSNBC['comNoms'])!=0
     assert type(resSNCPM)==dict
     assert len(resSNCPM)==2
     assert type(resSNCPM['comCodes'])==dict
     assert type(resSNCPM['comNoms'])==dict
     assert len(resSNCPM['comCodes'])!=0
     assert len(resSNCPM['comNoms'])!=0
     assert type(resSNIM)==dict
     assert len(resSNIM)==2
     assert type(resSNIM['comCodes'])==dict
     assert type(resSNIM['comNoms'])==dict
     assert len(resSNIM['comCodes'])!=0
     assert len(resSNIM['comNoms'])!=0
     print "DETECTION COMMUNAUTES OK"


def test_PR():

    IGgraph1=PageRank.getIGraph("MATCH (n:`andra-user`)<-[r:`andra-from`]-(t1:`andra-tweet`)-[re:`andra-retweet`]->(t2:`andra-tweet`)-[rel:`andra-from`]->(p:`andra-user`) WHERE t2.language="'"fr"'" RETURN n as nodeFrom,p as nodeTo",\
            grapheNeo,"192.168.1.75:7474","neo4j", "pass4dbse")
    IGgraph2=PageRank.getIGraph("MATCH (n:`andra-tweet`)-[re:`andra-retweet`]->(p:`andra-tweet`) RETURN n as nodeFrom,p as nodeTo",\
            grapheNeo,"192.168.1.75:7474","neo4j", "pass4dbse")

    assert type(IGgraph1)==igraph.Graph
    assert len(IGgraph1.vs)!=0
    assert len(IGgraph1.es)!=0

    assert type(IGgraph2)==igraph.Graph
    assert len(IGgraph2.vs)!=0
    assert len(IGgraph2.es)!=0

    PRank1=PageRank.Rank(IGgraph1,10)
    PRank2=PageRank.Rank(IGgraph2,10)

    assert type(PRank1)==dict
    assert len(PRank1['classement'])==10
    assert type(PRank1['classement'])==list
    assert type(PRank1['resPR'])==list
    assert len(PRank1['resPR'])!=0
    assert type(PRank2)==dict
    assert len(PRank1['classement'])==10
    assert type(PRank2['classement'])==list
    assert type(PRank2['resPR'])==list
    assert len(PRank2['resPR'])!=0
    print "PAGERANK OK"


def test_perfo():

    GraphElements=PerformanceFScore.importGraphElements(GraphFile,GTFile,path)

    assert type(GraphElements)==dict
    assert len(GraphElements)==4
    assert type(GraphElements['g'])==igraph.Graph
    assert len(GraphElements['g'].vs)!=0
    assert len(GraphElements['g'].es)!=0
    assert type(GraphElements['nbComParNoeud'])==dict
    assert len(GraphElements['nbComParNoeud'])!=0
    assert type(GraphElements['GT'])==str
    assert len(GraphElements['GT'])!=0
    assert type(GraphElements['nomIndice'])==dict
    assert len(GraphElements['nomIndice'])!=0

    Fscore=PerformanceFScore.calculFScore([1,6,4,8,5,12],[1,3,7,5,12])

    assert Fscore==0.545454545454545454545454545454545454545454545454545454545

    InfoIG=PerformanceFScore.Performance(GraphElements,"InfoIG",2,1000,path,path+'snap/',"FScore",14400)
    FastGreedy=PerformanceFScore.Performance(GraphElements,"FastGreedy",2,1000,path,path+'snap/',"FScore",14400)
    BigClam=PerformanceFScore.Performance(GraphElements,"BigClam",2,1000,path,path+'snap/',"FScore",14400)
    CPM=PerformanceFScore.Performance(GraphElements,"CPM",2,1000,path,path+'snap/',"FScore",14400)
    #InfoMap=PerformanceFScore.Performance(GraphElements,"InfoMap",2,1000,path,path+'snap/',"FScore")

    assert round(InfoIG,15)==0.000718390804598
    assert round(FastGreedy,15)==0.00065963060686
    assert BigClam==0
    assert CPM==0
    #assert round(InfoMap,15)==0.000718390804598

    InfoIG=PerformanceFScore.Performance(GraphElements,"InfoIG",2,1000,path,path+'snap/',"SimiNbCom",14400)
    FastGreedy=PerformanceFScore.Performance(GraphElements,"FastGreedy",2,1000,path,path+'snap/',"SimiNbCom",14400)
    BigClam=PerformanceFScore.Performance(GraphElements,"BigClam",2,1000,path,path+'snap/',"SimiNbCom",14400)
    CPM=PerformanceFScore.Performance(GraphElements,"CPM",2,1000,path,path+'snap/',"SimiNbCom",14400)
    #InfoMap=PerformanceFScore.Performance(GraphElements,"InfoMap",2,1000,path,path+'snap/',"SimiNbCom")

    assert InfoIG==-17.125
    assert FastGreedy==-5.125
    assert BigClam==-9.0
    assert CPM==-3.875
    #assert InfoMap==-49.5

    print "PERFORMANCE OK"


def test_influence():

    resIGFG=DetectionCommunautes.DetectionComIG(gTwitter,"FastGreedy")
    resIGIM=DetectionCommunautes.DetectionComIG(gTwitter,"InfoMap")
    resSNAPBC=DetectionCommunautes.DetectionComSNAP(gTwitter,"BigClam",path+'snap/',path,'outputGraph.txt','outputAlgo.txt')
    resSNAPIM=DetectionCommunautes.DetectionComSNAP(gTwitter,"InfoMap",path+'snap/',path,'outputGraph.txt','outputAlgo.txt')
    resSNAPCPM=DetectionCommunautes.DetectionComSNAP(gTwitter,"CPM",path+'snap/',path,'outputGraph.txt','outputAlgo.txt')

    NomComFG = InfluenceCommunautes.NomsCommunautes(resIGFG['membership'],gTwitter)
    NomComIM = InfluenceCommunautes.NomsCommunautes(resIGIM['membership'],gTwitter)

    assert type(NomComFG['comNoms'])==dict
    assert type(NomComFG['comNode'])==dict
    assert len(NomComFG['comNoms'])!=0
    assert len(NomComFG['comNode'])!=0
    assert type(NomComIM['comNoms'])==dict
    assert type(NomComIM['comNode'])==dict
    assert len(NomComIM['comNoms'])!=0
    assert len(NomComIM['comNode'])!=0

    IGgraph1=PageRank.getIGraph("MATCH (n:`andra-user`)<-[r:`andra-from`]-(t1:`andra-tweet`)-[re:`andra-retweet`]->(t2:`andra-tweet`)-[rel:`andra-from`]->(p:`andra-user`) WHERE t2.language="'"fr"'" RETURN n as nodeFrom,p as nodeTo",\
            grapheNeo,"192.168.1.75:7474","neo4j", "pass4dbse")
    PRank1=PageRank.Rank(IGgraph1,10)


    InfluFG=InfluenceCommunautes.InfluenceCommunautes(grapheNeo,gTwitter,PRank1['resPR'],NomComFG['comNode'],0.2,10)
    InfluIM=InfluenceCommunautes.InfluenceCommunautes(grapheNeo,gTwitter,PRank1['resPR'],NomComIM['comNode'],0.2,10)
    InfluBC=InfluenceCommunautes.InfluenceCommunautes(grapheNeo,gTwitter,PRank1['resPR'],resSNAPBC['comCodes'],0.2,10)
    InfluIMS=InfluenceCommunautes.InfluenceCommunautes(grapheNeo,gTwitter,PRank1['resPR'],resSNAPIM['comCodes'],0.2,10)
    InfluCPM=InfluenceCommunautes.InfluenceCommunautes(grapheNeo,gTwitter,PRank1['resPR'],resSNAPCPM['comCodes'],0.2,10)

    assert type(InfluFG)==dict
    assert type(InfluIM)==dict
    assert type(InfluBC)==dict
    assert type(InfluIMS)==dict
    assert type(InfluCPM)==dict


    InfluHTFG=InfluenceCommunautes.InfluenceHashtags(grapheNeo,NomComFG['comNode'],0.2)
    InfluHTIM=InfluenceCommunautes.InfluenceHashtags(grapheNeo,NomComIM['comNode'],0.2)
    InfluHTBC=InfluenceCommunautes.InfluenceHashtags(grapheNeo,resSNAPBC['comCodes'],0.2)
    InfluHTIMS=InfluenceCommunautes.InfluenceHashtags(grapheNeo,resSNAPIM['comCodes'],0.2)
    InfluHTCPM=InfluenceCommunautes.InfluenceHashtags(grapheNeo,resSNAPCPM['comCodes'],0.2)

    assert type(InfluHTFG)==dict
    assert type(InfluHTIM)==dict
    assert type(InfluHTBC)==dict
    assert type(InfluHTIMS)==dict
    assert type(InfluHTCPM)==dict
    assert len(InfluHTFG)==len(NomComFG['comNode'])
    assert len(InfluHTIM)==len(NomComIM['comNode'])
    assert len(InfluHTBC)==len(resSNAPBC['comNoms'])
    assert len(InfluHTIMS)==len(resSNAPIM['comNoms'])
    assert len(InfluHTCPM)==len(resSNAPCPM['comNoms'])

    InflueTweeFG=InfluenceCommunautes.InfluenceTweets(NomComFG['comNode'])
    InflueTweeIM=InfluenceCommunautes.InfluenceTweets(NomComIM['comNode'])
    InflueTweeBC=InfluenceCommunautes.InfluenceTweets(resSNAPBC['comCodes'])
    InflueTweeIMS=InfluenceCommunautes.InfluenceTweets(resSNAPIM['comCodes'])
    InflueTweeCPM=InfluenceCommunautes.InfluenceTweets(resSNAPCPM['comCodes'])

    assert type(InflueTweeFG)==dict
    assert type(InflueTweeIM)==dict
    assert type(InflueTweeBC)==dict
    assert type(InflueTweeIMS)==dict
    assert type(InflueTweeCPM)==dict
    assert len(InflueTweeFG)==len(NomComFG['comNode'])
    assert len(InflueTweeIM)==len(NomComIM['comNode'])
    assert len(InflueTweeBC)==len(resSNAPBC['comNoms'])
    assert len(InflueTweeIMS)==len(resSNAPIM['comNoms'])
    assert len(InflueTweeCPM)==len(resSNAPCPM['comNoms'])

    print "INFLUENCE OK"

test_Detec()
test_PR()
test_perfo()
test_influence()
