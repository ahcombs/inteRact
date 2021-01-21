"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Cooperative Robots  test file 3
Author: Jesse Hoey  jhoey@cs.uwaterloo.ca   http://www.cs.uwaterloo.ca/~jhoey
December 2013
Use for research purposes only.
Please do not re-distribute without written permission from the author
Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by the Natural Sciences and Engineering Council of Canada (NSERC).
see README for details
----------------------------------------------------------------------------------------------"""
from pomcp import *
import numpy as NP
import threading
import time

#simple threading class so we can do two things at once
class myThread (threading.Thread):
    def __init__(self, threadID, name, tAgent):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.tAgent = tAgent
        self.result = None
    def run(self):
        print "Starting " + self.name
        self.result = self.tAgent.get_next_action()
        print "Exiting " + self.name

def printMeanState(bs):
    meanstate=0
    numsamps=0
    meanfc=NP.zeros((1,3))
    meanturn=0
    for thestate in bs:
        numsamps=numsamps+1
        meanstate = meanstate+thestate.y
        meanturn=meanturn+thestate.turn
        meanfc = meanfc + thestate.fc
    print "meanstate for ",numsamps, " samples is: ",meanstate/numsamps
    print "mean identity estimate is : ",NP.array(meanfc)/numsamps
    print "mean turn estimate is : ",meanturn/numsamps

    
def invert_turn(turn):
    if turn=="agent":
        return "client"
    else:
        return "agent"

#this is like testpomcp3, but now the turn is truly represented by the self.turn
#and we allow for a partially observable turn variable
#and "x" now models the position of the other agent, 
#and there is no push/notpush action/state at all
class State(object):

    turnnames=["agent","client"]

    def __init__(self,fc,y,x,turn,weight):
        self.fc=fc
        self.y=y   #position of the agent
        self.x=x   #position of the other agent
        self.turn=turn
        self.weight=weight

    def __add__(self,other):
        return State(self.fc+other.fc,self.y+other.y,self.x+other.x,self.turn+other.turn,self.weight+other.weight)
    def __mul__(self,other):
        return State(self.fc*other.fc,self.y*other.y,self.x*other.x,self.turn+other.turn,self.weight+other.weight)
    def __div__(self,other):
        #hmmm - how can this work? 
        return State(NP.divide(self.fc,other),NP.divide(self.y,other),NP.divide(self.x,other),NP.divide(self.turn,other),NP.divide(self.weight,other))
    def __iadd__(self,other):
        self.fc=NP.add(self.fc,other.fc)
        self.y=NP.add(self.y,other.y)
        self.x=NP.add(self.x,other.x)
        self.turn = NP.add(self.turn,other.turn)
        self.weight=NP.add(self.weight,other.weight)
        return self

    def invert_turn(self):
        return self.turnnames.index(invert_turn(self.get_turn()))

    def get_turn(self):
        return self.turnnames[int(round(self.turn))]


    def get_copy(self):
        return State(self.fc,self.y,self.x,self.turn,self.weight)

    def apply_weight(self):
        return State(NP.dot(self.fc,self.weight),NP.dot(self.y,self.weight),NP.dot(self.x,self.weight),NP.dot(self.turn,self.weight),self.weight)

    #add some noise to samples ("roughening" - see Gordon et al. 1993 or Chapt. 10 of Nando's book)
    def roughen(self,noiseVector):
        nois=(2*NP.random.random((1,3))-1.0)*noiseVector
        self.fc = self.fc + nois.flatten(1)
        #nois=(2*NP.random.random((1,9))-1.0)*noiseVector
        #self.tau = self.tau + nois.flatten(1)


    def print_val(self):
        print 50*"~"
        print "fc: ",self.fc
        print "y: ",self.y
        print "x: ",self.x
        print "turn: ",self.turn
        print "weight: ",self.weight
        print 50*"~"

#A corobot models its own position (y) and the position of the other robot (x) 
#as well as the identity of the other robot (fc), the turn (turn) 
#it has a self identity self.identity 
#it receives observations about the position of the other robot, about its own position 
#and about the identity of the other robot.  Corobots are forced to reveal their true
#identities to the other robot at each step.
class CoRobot(object):
    def __init__(self,*args,**kwargs):
        self.goalstate=kwargs.get("goal",10.0)
        self.timeout=kwargs.get("timeout",1.0)
        self.oracleSigma=kwargs.get("oraclesig",1.0)
        self.perfectOracle=kwargs.get("poracle",False)
        self.obsres=kwargs.get("obsres",1.0)
        self.actres=kwargs.get("actres",1.0)
        self.numcact=kwargs.get("numcact",2)
        self.firstturn = State.turnnames.index(kwargs.get("turn","agent"))
        self.identity = kwargs.get("identity",[0,0,0])  #E-P-A

        #only used in cases where a perfect oracle is used for testing purposes
        self.clientidentity = kwargs.get("clientidentity",[0,0,0])  #E-P-A

        self.rewsigma=2.5
        self.N=1000
        self.rough = self.N**(-1.0/3.0)
        self.dyn_noise=0.1
        self.obs_noise=0.5
        self.id_noise=0.1
        self.id_obs_noise=1.0
        self.discount_factor=0.9

        
        self.beliefSimilarity = -1.0

        #params of sigmoid function
        self.sigidA = 1.0
        self.sigidP = 1.0


        #negative identities will strive towards negative goals
        self.oracleMeanValue=0.678
        self.oracleMean=self.oracleMeanValue
        if self.identity[0] < 0:
            self.oracleMean = -1.0*self.oracleMeanValue
        

        self.initialise()
        self.POMCP_initialise()

    def POMCP_initialise(self):
        #should add actres
        self.pomcp_agent=POMCP(cval=1.0,numcact=self.numcact,numdact=1,obsres=self.obsres,actres=self.actres,timeout=self.timeout)  
        self.pomcp_agent.POMCP_eval(self.beliefState,200,self)

    def printParams(self):
        self.pomcp_agent.printParams()

    def initialise(self):
        #initial belief set
        #each samples is a tuple (continuous_state,discrete_state,weight)
        #we draw all the continuous dimensions independently, although could get some covariances and do them 
        #with multivariate_normal just as well
        ySamples = NP.random.normal(0,2.0,self.N)
        xSamples = NP.random.normal(0,2.0,self.N)
        id_eSamples = NP.random.normal(0,4.0,self.N)
        id_pSamples = NP.random.normal(0,4.0,self.N)
        id_aSamples = NP.random.normal(0,4.0,self.N)
        self.beliefState=[]
        index=0
        for (ysample,xsample) in zip(ySamples,xSamples):
            self.beliefState.append(State([id_eSamples[index],id_pSamples[index],id_aSamples[index]],ysample,xsample,self.firstturn,1.0))   ##hard-coded agent goes first??
            index += 1

        return self.beliefState


    def getPossibleNextActions(self):
        return self.pomcp_agent.getPossibleNextActions()
        

    def get_next_action(self):
        return self.pomcp_agent.POMCP_search(self.beliefState,self,self.timeout)

    def sigmoid(self,id1,id2):
        idP = id1[1]-id2[1]
        idA = id1[2]-id2[2]
        return 1.0/(1+math.exp(-idP/self.sigidP-idA/self.sigidA))
    

    #generate new sample and return it
    #these functions are  used in the POMCP class to call the agent and get a sample
    def propagateSample(self,state,action):

        #power differential
        pa = self.sigmoid(self.identity,state.fc)

        #agent action
        yaction    = action[0]
  
        #client action depends on the power differential
        if NP.random.random() > pa:
            #draw from client - client will do his own thing based on his identity valence
            if state.fc[0] > 0:
                xaction = self.oracleMeanValue
            else:
                xaction = -1.0*self.oracleMeanValue
        else:
            #draw from agent - client will do the thing based on agent identity
            #could make it so the client just moves towards the agent as well here
            if self.identity[0] > 0:
                xaction = self.oracleMeanValue
            else:
                xaction = -1.0*self.oracleMeanValue
        #this is our prediction noise for client behaviours
        xaction = xaction + NP.random.normal(0,self.oracleSigma,1)
                
        #use same noise terms for both agents for simplicity
        newy = state.y + yaction  + NP.random.normal(0,self.dyn_noise,1)
        newx = state.x + xaction  + NP.random.normal(0,self.dyn_noise,1)
        newyobs = newy + NP.random.normal(0,self.obs_noise,1)
        newxobs = newx + NP.random.normal(0,self.obs_noise,1)

        if state.turn==0:  #agent
            #its client turn if client dominates - remove this for now
            if True or NP.random.random() > pa:  
                newturnobs=1
                state.turn=1
            else:
                #stays agent turn
                newturnobs=0
                state.turn=0
        else:
            #stays client turn
            if False and NP.random.random() > pa:   #remove random turn switching for now
                newturnobs=1
                state.turn=1
            else:
                newturnobs=0
                state.turn=0
        
        newreward=0.0
        if abs(state.x-state.y)<1.0:  #agents are close enough
            if self.identity[0] > 0:
                newreward = (10.0*math.exp(-((state.y-self.goalstate)/self.rewsigma)**2.0)
                             + 3.0*math.exp(-((state.y+self.goalstate)/self.rewsigma)**2.0))
            else:
                newreward = (10.0*math.exp(-((state.y+self.goalstate)/self.rewsigma)**2.0)
                             + 3.0*math.exp(-((state.y-self.goalstate)/self.rewsigma)**2.0))
        
        newfc = state.fc + NP.random.normal(0,self.id_noise,3)
        newfcobs = state.fc + NP.random.normal(0,self.id_obs_noise,3)
        return (State(newfc,newy,newx,state.turn,state.weight),(newyobs,newxobs,newturnobs,newfcobs[0],newfcobs[1],newfcobs[2]),newreward)
        
    def updateBelief(self,action,observation,verb=False):
        newsamples=[]
        for state in self.beliefState:
            #print 100*"KKK"
            #state.print_val()
            (newstate,newobs,newreward)=self.propagateSample(state,action)
            #newstate.print_val()
            #print newobs
            #print observation
            newstate.weight = normpdf_old(observation[0],newstate.y,self.obs_noise)  
            newstate.weight += normpdf_old(observation[1],newstate.x,self.obs_noise) 
            #print newstate.weight
            newstate.weight+=normpdf_old(observation[3:],newstate.fc,self.id_obs_noise) 
            
            #print newstate.weight
            newstate.weight = math.exp(newstate.weight) #don't need this because its all deterministic self.evalSampleXvar(newstate,observation

            if not newstate.turn == observation[2]:
                newstate.weight = 0.0

            #print newstate.weight
            newsamples.append(newstate)
            #if newstate.weight > 1e-200:
            #    print "weight: ",normpdf_old(observation[0],newobs[0],self.obs_noise)
            #    print normpdf_old(observation[3:],newobs[3:],self.id_obs_noise)

        if verb:
            for state in newsamples:
                state.print_val()
            
        
        #draw unweighted set
        self.beliefState=drawSamples(newsamples,self.N)
        if verb:
            for state in self.beliefState:
                state.print_val()

        #possibly roughen samples
        if self.rough>0:
            self.roughenSamples(self.beliefState)

        return self.beliefState
                            
            
            #add some noise to samples ("roughening" - see Gordon et al. 1993 or Chapt. 10 of Nando's book)
    def roughenSamples(self,samples):
        noiseVector=(NP.ones((3,1))*self.rough).flatten(1)
        map(lambda x: x.roughen(noiseVector),samples)


    #rollout is a boolean flag  - if we are doing a rollout, then
    #this sample function computes and returns the rollout policy
    def oracle(self,state,rollout=False,samplenum=0):
        #this should be in an ACT subclass.  In this parent class, we should draw a rangom sample
        #but I'm leaving this here for now so that I don't start two files at the same time
        #continuous component of action
        a=NP.array([0.0])
        #use the sigmoid computation
        #if ps>0.5 it means the agent is in control
        pa = self.sigmoid(self.identity,state.fc)
        if pa > 0.5:
            omean = self.oracleMean
        elif state.fc[0]>0:
            omean = self.oracleMeanValue
        else:
            omean = -1*self.oracleMeanValue

        #the base action for movement is biased towards the client identity
        a=NP.random.normal(omean,self.oracleSigma,1)
            
        if self.perfectOracle:
            pa = self.sigmoid(self.identity,self.clientidentity)
                
            if pa > 0.5:
                if self.identity[0] > 0:
                    realGoalDiff = self.goalstate-state.y
                else:
                    realGoalDiff = -1.0*self.goalstate-state.y
            elif self.clientidentity[0] > 0:
                realGoalDiff = self.goalstate-state.y
            else:
                realGoalDiff = -1.0*self.goalstate-state.y
            a[0] = realGoalDiff



        return (a,)

    def get_num_actions(self,state):
        if state.turn==0:
            return (self.pomcp_agent.numDiscActions,1)  # was: self.pomcp_agent.numContActions), but now we add incrementally
        else:
            return (1,1)
    

    #an obsSet is a combination of two elements - 
    #First, a N-d continuous vector and 
    #Second, the the propositional observation,
    #which is actually class dependent
    #needs to be generalised to it takes any observation and can be moved into pomcp.py perhaps
    def getMeanObs(self,obsSet):
        
        if obsSet:
            avgyobs = NP.array(obsSet[0][0])
            avgeobs = NP.array(obsSet[0][3])
            avgpobs = NP.array(obsSet[0][4])
            avgaobs = NP.array(obsSet[0][5])
            avgxobs = obsSet[0][1]
            avgtobs = obsSet[0][2]
            for o in obsSet[1:]:
                avgyobs = avgyobs+o[0]
                avgeobs = avgeobs+o[3]
                avgpobs = avgpobs+o[4]
                avgaobs = avgaobs+o[5]
                avgxobs = avgxobs+o[1]
                avgtobs = avgtobs+o[2]
            avgyobs = avgyobs/len(obsSet)
            avgeobs = avgeobs/len(obsSet)
            avgpobs = avgpobs/len(obsSet)
            avgaobs = avgaobs/len(obsSet)
            avgxobs = avgxobs/len(obsSet)
            avgtobs = avgtobs/len(obsSet)

            return (avgyobs,avgxobs,avgtobs,avgeobs,avgpobs,avgaobs)
        else:
            return []

    def contObsDist(self,obs1,obs2):
        return  (math.sqrt(raw_dist(obs1[0],obs2[0]) +
                           raw_dist(obs1[3],obs2[3]) +   #left out x? 
                           raw_dist(obs1[4],obs2[4]) +
                           raw_dist(obs1[5],obs2[5])))
        

    #another one that is class dependent
    #this is not used anymore
    def observationMatch(self,obs1,obs2,resolv):
        theans = ((obs1[2]==obs2[2]) and self.contObsDist(obs1,obs2) < resolv)
        print self.contObsDist(obs1,obs2),resolv,theans
        return theans

    def bestObservationMatch(self,obs,obsSet,obsSetData):
        firstone=True
        bestdist=-1
        besto=-1
        for oindex in range(len(obsSet)):
            o=obsSet[oindex]
            if obs[2]==o[2]:
                odist = self.contObsDist(obs,o)
                if firstone or odist<bestdist:
                    firstone=False
                    besto=oindex
                    bestdist=odist
            #else: print obs,o  #o[1] is the problem - seems to be a real number, not 0/1
        #print (besto,bestdist,len(obsSet),len(obsSetData))

        return (besto,bestdist)

    #check to see if action is in actionSet (to within actres resolution)
    def hasAction(self,actionSet,action,actres):
        for a1 in actionSet:
            if math.sqrt(raw_dist(action[0],a1[0])) < actres:       
                return True
        return False

    #distance bewteen two actions - Negative means infinite
    def actionDist(self,a1,a2):
        return math.sqrt(raw_dist(a1[0],a2[0]))

    #distance between two obwervatoins - negative means infinite
    def observationDist(self,obs1,obs2):
        if obs1[2]==obs2[2]:
            return self.contObsDist(obs1,obs2)
        else:
            return -2

rseed = NP.random.randint(0,382948932)
#rseed=75639353

print "random seeed is : ",rseed
NP.random.seed(rseed)


trueClientId = [-2.0,2.0,2.0]
trueAgentId = [2.0,-2.0,-2.0]
trueAgentTurn="agent"
trueClientTurn="client"
trueY = 0.0
trueX = 0.0
trueDynNoise = 0.1
trueObsNoise = 0.1
trueIdObsNoise = 0.1
trueGoal = 10.0
trueRewSigma = 2.5

obsres = 5.0
actres = 1.0
numcact = 10

pomcptimeout=5.0

osig=1.0

randomids = False
poracle = False
#if negative, run forever
numiterations=-1  

#if True, will pause the simulation for search tree exploratations
doInteractive = False

#get user inputs if there
if len(sys.argv) > 1:
    pomcptimeout=float(sys.argv[1])
if len(sys.argv) > 2:
    osig=float(sys.argv[2])
if len(sys.argv) > 3:
    numiterations=int(sys.argv[3])
if len(sys.argv) > 4:
    numcact=int(sys.argv[4])
if len(sys.argv) > 5:
    actres=float(sys.argv[5])
if len(sys.argv) > 6:
    aa = sys.argv[6].strip()
    randomids=not (aa=="False" or aa=="F" or aa=="0" or aa=="f")
if len(sys.argv) > 7:
    aa = sys.argv[7].strip()
    poracle=not (aa=="False" or aa=="F" or aa=="0" or aa=="f")




print "running with "
print "pomcp timeout : ",pomcptimeout
print "oracle sigma  : ",osig
print "num iterations: ",numiterations
print "random ids?: ",randomids
print "perfect oracle?: ",poracle


if randomids:
    trueClientId = NP.random.normal(0,2.0,3)
    trueAgentId = NP.random.normal(0,2.0,3)

print "client id: ",trueClientId
print "agent id:  ",trueAgentId
    



totalreward=0.0

iteration=0

#true client id is passed to each agent, but only used by the oracle when poracle is true for tests/comparisons
testAgent = CoRobot(goal=trueGoal,x=trueX,identity=trueAgentId,clientidentity=trueClientId,turn=trueAgentTurn,timeout=pomcptimeout,obsres=obsres,actres=actres,numcact=numcact,oraclesig=osig,poracle=poracle)

testClient = CoRobot(goal=trueGoal,x=trueX,identity=trueClientId,clientidentity=trueAgentId,turn=trueClientTurn,timeout=pomcptimeout,obsres=obsres,actres=actres,numcact=numcact,oraclesig=osig,poracle=poracle)

beliefStateAgent = testAgent.initialise()
beliefStateClient = testClient.initialise()

testAgent.printParams()
testClient.printParams()



while numiterations<0 or iteration<numiterations:
    
    print "*** agent mean of belief state:",20*"*"
    printMeanState(beliefStateAgent)
    print 50*"*"
    print "planning next move (Agent)..."
    print


    if False:
        print "agent belief state samples: "
        for b in beliefStateAgent:
            b.print_val()
            a=testAgent.oracle(b)
            print "action from oracle: ",a
            pa = testAgent.sigmoid(testAgent.identity,b.fc)
            print "pa is: ",pa
            print "self id: ",testAgent.identity
            print "estimate of other id: ",b.fc

    print "*** client mean of belief state:",20*"*"
    printMeanState(beliefStateClient)
    print 50*"*"
    print "planning next move (Client)..."
    print



    #figure out what the agent should do based on the beliefState
    #do this in threads so it goes twice as fast
    agentThread = myThread(1, "agentThread", testAgent)
    clientThread = myThread(2, "clientThread", testClient)

    agentThread.start()
    clientThread.start()

    while agentThread.isAlive() or clientThread.isAlive():
        print ".",
        time.sleep(1)  #seconds
    
    (avalue,agent_action)  = agentThread.result
    (cvalue,client_action)  = clientThread.result


    #testAgent.pomcp_agent.ucTree.print_val("plantreeAgent"+str(iteration)+".txt")
    #testClient.pomcp_agent.ucTree.print_val("plantreeClient"+str(iteration)+".txt")

    print "agent actions considered: "
    testAgent.getPossibleNextActions()
    print "agent action is: ",agent_action 

    print "client actions considered: "
    testClient.getPossibleNextActions()
    print "client action is: ",client_action 

    trueTurn = trueAgentTurn
    yaction = agent_action[0]
    xaction = client_action[0]
    
    trueY = trueY + yaction  + NP.random.normal(0,trueDynNoise,1)
    trueX = trueX + xaction  + NP.random.normal(0,trueDynNoise,1)

    trueYobs = trueY + NP.random.normal(0,trueObsNoise,1)
    trueXobs = trueX + NP.random.normal(0,trueObsNoise,1)

    trueAgentTurn = invert_turn(trueAgentTurn)
    trueClientTurn = invert_turn(trueClientTurn)

    #each agent gets to observe the other's identity, with a bit of added noise
    agentIdObs = testClient.identity+NP.random.normal(0,trueIdObsNoise,3)
    clientIdObs = testAgent.identity+NP.random.normal(0,trueIdObsNoise,3)

    #each agent gets to observe the true Turn, and a noisy version of Y, X
    newAgentObs = (trueYobs,trueXobs,State.turnnames.index(trueAgentTurn),agentIdObs[0],agentIdObs[1],agentIdObs[2])
    newClientObs = (trueXobs,trueYobs,State.turnnames.index(trueClientTurn),clientIdObs[0],clientIdObs[1],clientIdObs[2])

    print 100*"-"
    print
    print "true new state is: ",
    print "Y: ", trueY,  " X: ",trueX
    print " observation (agent): ",newAgentObs
    print " observation (client): ",newClientObs
    print
    print 100*"-"
    
    newreward  = 0.0

    ypgoaldist=math.exp(-((trueY-trueGoal)/trueRewSigma)**2.0)
    yngoaldist=math.exp(-((trueY+trueGoal)/trueRewSigma)**2.0)

    xpgoaldist=math.exp(-((trueX-trueGoal)/trueRewSigma)**2.0)
    xngoaldist=math.exp(-((trueX+trueGoal)/trueRewSigma)**2.0)
    

    if abs(trueY-trueX)<1.0:
        if trueClientId[0]>0:
            newreward = newreward + 10.0*xpgoaldist
            newreward = newreward + 3.0*xngoaldist
        else:
            newreward = newreward + 3.0*xpgoaldist
            newreward = newreward + 10.0*xngoaldist
            
        if trueAgentId[0]>0:
            newreward = newreward + 10.0*ypgoaldist
            newreward = newreward + 3.0*yngoaldist
        else:
            newreward = newreward + 3.0*ypgoaldist
            newreward = newreward + 10.0*yngoaldist

    if doInteractive:
        askq=raw_input("I'm about to update the belief states and prune the tree, but maybe you want to interactively explore the tree? (enter means no, anything else is yes):")
        if not askq == "":
            testAgent.pomcp_agent.POMCP_interactiveExplorePlanTree()
    
    #the agent's belief about that state
    print "upating agent belief ..."
    beliefStateAgent = testAgent.updateBelief(agent_action,newAgentObs)
    
    print "upating client belief ..."
    beliefStateClient = testClient.updateBelief(client_action,newClientObs)
    

    #print 10*"-.","top level root node: "
    #testAgent.pomcp_agent.POMCP_printLevelVal()

    (besta,besto,bestd) = testAgent.pomcp_agent.POMCP_pruneTree(agent_action,newAgentObs,testAgent)

    if besta < 0 or besto < 0:
        #we did not a match for some reason - the only we can do is reset
        print 200*"r"+"eset!!!"
        testAgent.POMCP_initialise()
        #this is not right - 
        #testAgent=TestPOMCP()

    #print 10*"-.","top level root node: "
    #testClient.pomcp_agent.POMCP_printLevelVal()

    (besta,besto,bestd) = testClient.pomcp_agent.POMCP_pruneTree(client_action,newClientObs,testClient)
    

    if besta < 0 or besto < 0:
        #we did not a match for some reason - the only we can do is reset
        print 200*"r"+"eset!!!"
        testClient.POMCP_initialise()

    #print 10*"-.","new top level root node: "
    #testAgent.pomcp_agent.POMCP_printLevelVal()
    
    
    print "reward obtained: ",newreward, " so far: ",totalreward
    
    
    totalreward=totalreward+newreward

    iteration=iteration+1

print "final reward obtained (in ",iteration," iterations): ",totalreward

    
