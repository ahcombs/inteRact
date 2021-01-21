"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Discrete Tutor Subclass of Agent class
Author: Jesse Hoey  jhoey@cs.uwaterloo.ca   http://www.cs.uwaterloo.ca/~jhoey
September 2013
Use for research purposes only.
Please do not re-distribute without written permission from the author
Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by the Natural Sciences and Engineering Council of Canada (NSERC).
use python2.6
see README for details
----------------------------------------------------------------------------------------------"""

from bayesact import *


class DiscreteTutoringAgent(Agent):
    def __init__(self,*args,**kwargs):

        super(DiscreteTutoringAgent, self).__init__(self,*args,**kwargs)
        #x for the DiscreteTutoringAgent is a triple: 
        #the first value is the turn (always this way for any Agent)
        #the second value
        #is the difficulty level (0-2) of the previous exercise, and
        #the third is the skill level of the student
        #skill levels
        self.xvals=["easy","medium","hard"]
        #probability distribution over skill levels
        self.px=kwargs.get("initx",[0.2,0.6,0.2])
        #basic distributions for deflection of 1
        self.ppx=kwargs.get("ppx",[[0.9,0.1,0.0],[0.05,0.9,0.05],[0.0,0.1,0.9]])
        #prob. of failure/success for each difficulty level: easy, medium, hard
        #for each value of x[2]:   easy,med,hard
        self.of=kwargs.get("of",[[[0.1,0.9],[0.01,0.99],[0.001,0.999]],#x[1]=easy
                                 [[0.5,0.5],[0.1,0.9],[0.01,0.99]],#x[1]=med
                                 [[0.9,0.1],[0.5,0.5],[0.1,0.9]]])#x[1]=hard
        self.goalx=kwargs.get("goalx",2)


    def print_params(self):
        Agent.print_params(self)
        print "px: ",self.px
        print "ppx: ",self.ppx
        print "goal x: ",self.goalx

    #The only difference between a TutoringAgent and an Agent
    #is dynamics of x and reward
    def initialise_x(self,initx=None):
        if not initx:
            initpx=self.px
            initturn=State.turnnames.index("agent")
        else:
            initpx=initx[1]
            initturn=initx[0]
        initskill = list(NP.random.multinomial(1,initpx)).index(1)
        return [initturn,0,initskill]

    #implement a default policy where we look at the client's current abilities  (average x value)
    #and pick a difficulty level corresponding to P(x) - the distribution over x for client
    #select an action at the difficulty level corresponding to the client's ability
    #and occasionally one that is one level higher
    #avgstate=sampleMean(self.samples)  #will be a real number from 0-2
    def get_prop_action(self,state,rollout=False,samplenum=0):
        if self.use_pomcp:
            if rollout:
                paab = NP.random.randint(self.numDiscActions) 
            else:
                paab = samplenum%self.numDiscActions
        else:
            skill = state.x[2]
            paab=int(round(skill))
            if NP.random.random()>0.9:
                paab=min(2,paab+1)
        return paab

    #was greedy_action_select
    #selects an action greedily by a
    def get_default_action(self,state,paab=None):
        return self.get_greedy_action(state,paab)

    def sampleXObservation(self,state):
        if state.get_turn()=="agent":
            return [state.x[0],list(NP.random.multinomial(1,self.of[state.x[1]][state.x[2]])).index(1)]
        else:
            return [state.x[0],0]


    #aab is the affective part of the action, paab is the propositional part
    def sampleXvar(self,f,tau,state,aab,paab):
        if state.get_turn()=="client":
            #compute deflection between f and tau
            #x moves towards goalx inversely proportionally to 
            #the deflection
            D=NP.dot(f-tau,f-tau)
            #as deflection grows, the distribtuion is more likely to favor staying the same or regressing, not progressing
            xppx=NP.asarray(self.ppx[state.x[2]])
            #current and all previous ones as wellget increased by deflection
            for y in range(state.x[2]+1):
                xppx[y] = xppx[y]*(D/4.0)  #this is likely incorrect - why did I do this with a *?
            #y=x[1]
            xppx = xppx/float(xppx.sum())
            #sample from this to get the new x value
            new_x=list(NP.random.multinomial(1,xppx)).index(1)
            #print xppx
            #print new_x
            return [state.invert_turn(),state.x[1],new_x]
        else:
            return [state.invert_turn(),paab,state.x[2]]
    
    #this must be in the class
    def evalSampleXvar(self,sample,xobs,oldsample=[]):
        #this nasty little hack is because I'm calling update(..) to update the agent
        #in tutorgui2.py, which makes a default assumption that on agent turn, no observations
        #are received. 
        #if xobs==[] or not turn=="client":
        #    return 1.0
        #only when it was the client turn (so it is now agent turn) do we use the observation function
        if sample.get_turn()=="agent":
            if xobs[0]==sample.x[0]:
                #print sample.x[0],sample.x[1],xobs,self.of[sample.x[0]][sample.x[1]][xobs]
                return self.of[sample.x[1]][sample.x[2]][xobs[1]]
            else: 
                return 0.0
        else:
            return 1.0   #ignored x observation

    def reward(self,sample,action=None):
        # a generic deflection-based reward that can be removed eventually
        fsample=sample.f.transpose()
        freward = -1.0*NP.dot(sample.f-sample.tau,sample.f-sample.tau)
        # a state-based reward that favors states closer to the goal
        xreward = -1.0*(sample.x[2]-self.goalx)**2

        return freward+xreward


    #----------------------------------------------------------------------------------
    #POMCP functions
    #----------------------------------------------------------------------------------

    def oracle(self,state,rollout=False,samplenum=None):
        #continuous component is just a 3D EPA vector which is the null value on client turn
        #the discrete component of the action is class dependent
        #here, if it is the client's turn, we simply use the null action for both 
        #and in any case, the propositional action is always the same (0 = null)
        (a,paab)  = self.get_null_action()

        if state.get_turn()=="agent":
            #the first sample returned is the default (ACT) action
            if samplenum==0:
                (a,paab)=self.get_default_action(state)
                paab = 0
            else:
                #afterwards, it is a random selection
                #and paab gets set to a sequence dependent on sampleNum
                #isiga_pomcp is much smaller than isiga
                usesiga=self.isiga_pomcp
                #when doing a rollout, want to cast a wider net? 
                if rollout:
                    usesiga=self.isiga
                (tmpfv,wgt,tmpH,tmpC)=sampleFvar(fvars,tvars,self.iH,self.iC,usesiga,self.isigf_unconstrained_b,state.tau,state.f,"agent",[],[])

                a=map(lambda x: float(x), tmpfv[3:6])

                #loop over all discrete actions, or get randomly if doing a rollout
                paab = self.get_prop_action(state,rollout,samplenum)


        return (a,paab)

    #returns the closest observation match to obs in obsSet
    #this needs to be in this class, as the comparison depends on the structure of the observation vector
    def bestObservationMatch(self,obs,obsSet,obsSetData):
        firstone=True
        bestdist=-1
        besto=-1
        for oindex in range(len(obsSet)):
            o=obsSet[oindex]
            if obs[1][0]==o[1][0] and obs[1][1]==o[1][1]:
                odist = math.sqrt(raw_dist(obs[0],o[0]))
                if firstone or odist<bestdist:
                    firstone=False
                    besto=oindex
                    bestdist=odist
        return (besto,bestdist)
        
    def observationDist(self,obs1,obs2):
        if obs1[1][0] == obs2[1][0] and obs1[1][1]==obs2[1][1]:
            return math.sqrt(raw_dist(obs1[0],obs2[0]))
        else:
            return -2

