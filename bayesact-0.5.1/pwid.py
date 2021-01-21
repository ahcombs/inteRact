"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Person with Dementia subclass of Agent class
Based on the Assistant subclass
Author: Jesse Hoey  jhoey@cs.uwaterloo.ca   http://www.cs.uwaterloo.ca/~jhoey
September 2013
Use for research purposes only.
Please do not re-distribute without written permission from the author
Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by the Natural Sciences and Engineering Council of Canada (NSERC).
use python
see README for details
----------------------------------------------------------------------------------------------"""

#also change to bayesact in bayesactassistant.py
from bayesact import *
import copy

class PwD(Agent):
    def __init__(self,*args,**kwargs):

        super(PwD, self).__init__(self,*args,**kwargs)
        #x for the Assistant is a triple: 
        #the first is the turn (required)
        #the second value
        #is the planstep (0 to N-1) and the third
        # is the binary awareness of the client 0/1

        #dictionary defining the dynamics of the state and behaviour
        self.nextPsDict = kwargs.get("nextpsd",{0:([1.0],[1]),1:([1.0],[1])})
        self.nextBehDict = kwargs.get("nextbd",{0:{0:([1.0],[0]),1:([1.0],[1])},1:{0:([1.0],[1]),1:([1.0],[1])}})
        self.num_plansteps=len(self.nextPsDict)
        self.num_behaviours=len(self.nextBehDict)


        #initial awarness distribution
        self.px = [0.3,0.7]
        
        #dynamics of behaviours as a function of integer deflections from 0-9
        #defb[d] gives the probability that a person will move to the next planstep 
        #if they are aware, they were given a prompt and the deflection has value d
        #self.defb = [1.0,0.99,0.95,0.7,0.5,0.3,0.2,0.1,0.05,0.005]

        #turned these downwards after adding behaviours - seems deflections were not 
        #as high, which may be just because of added uncertainty
        self.defb = [1.0,0.99,0.9,0.3,0.2,0.1,0.05,0.005,0.001,0.0001]

        #these are the numbers for a person losing awareness spontaneously, without a prompt
        #defbnp[d] gives the probability that a person will move to the next planstep
        #if they are aware and the deflection has value d
        #self.defbnp = [0.99,0.95,0.7,0.5,0.3,0.2,0.1,0.05,0.005,0.001]
        #self.defbnp = [0.8,0.6,0.5,0.3,0.2,0.1,0.05,0.005,0.002,0.001]
        #downtuned versions
        self.defbnp = [0.8,0.6,0.3,0.2,0.1,0.05,0.005,0.002,0.001,0.0001]

        self.currPlanStep=0
        self.currBehaviour=0
        self.currAwareness=list(NP.random.multinomial(1,self.px)).index(1)

        self.lastPrompt = None

    def print_params(self):
        Agent.print_params(self)
        print "num plansteps: ",self.num_plansteps
        print "current (initial) awareness: ",self.currAwareness
        print "ps dynamics: ",self.nextPsDict

    def print_state(self):
        print "pwid current awareness: ",self.currAwareness
        print "pwid current planstep: ",self.currPlanStep
        print "pwid current lastPrompt: ",self.lastPrompt
        
    def is_done(self):
        return self.currPlanStep == self.num_plansteps-1
    #get the next planstep to take according to a plan-graph
    #def getNextPlanStep(self):
    #    return min(self.num_plansteps,self.currPlanStep+1)   #in general, could be next step according to some plan-graph

    #old version before moving to behaviours
    #def getNextPlanStep(self):
    #    newps = self.nextPsDict[self.currPlanStep][1][list(NP.random.multinomial(1,self.nextPsDict[self.currPlanStep][0])).index(1)] 
    #    return newps

    #get the expected next planstep
    def getNextPlanStep(self,planstep,behaviour):
        newps=self.nextPsDict[planstep][behaviour][1][list(NP.random.multinomial(1,self.nextPsDict[planstep][behaviour][0])).index(1)]
        return newps 

    #get the expected next behaviour
    def getNextBehaviour(self,planstep):
        newbeh=self.nextBehDict[planstep][1][list(NP.random.multinomial(1,self.nextBehDict[planstep][0])).index(1)]
        return newbeh 

    #get the expected next behaviour when prompted - biased towards the prompt 
    def getNextBehaviourPrompted(self,planstep,promptedBeh):
        nextBehaviours=copy.copy(self.nextBehDict[planstep][1])
        pNextBehaviours=copy.copy(self.nextBehDict[planstep][0])  #distribution over next behaviours
        
        if promptedBeh in nextBehaviours:
            #rescale the distribution
            pNextBehaviours[nextBehaviours.index(promptedBeh)] += 0.5  #effect constant
        else:
            nextBehaviours += [promptedBeh]   #add this new behaviour to the list if not there
            pNextBehaviours += [0.0]
            pNextBehaviours[nextBehaviours.index(promptedBeh)] += 0.1  #effect constant
        
        
        #renormalize the distribution
        sump = sum(pNextBehaviours)
        pNextBehaviours = map(lambda x: x/sump,pNextBehaviours)
        
        print "sampling from: ",pNextBehaviours," behaviours: ",nextBehaviours

        newbeh=nextBehaviours[list(NP.random.multinomial(1,pNextBehaviours)).index(1)]
        return newbeh 
            
    #The only difference between an Assistant and an Agent
    #is dynamics of x and reward
    def initialise_x(self,initx):
        if not initx:
            initpx=self.px
            initturn=State.turnnames.index("client")  #hard-coded client start??
            print "initpx: ",initpx, "initturn: ",initturn
        else:
            initpx=initx[1]
            initturn=State.turnnames.index(initx[0])
        #draw a sample from initx over awareness
        initx=list(NP.random.multinomial(1,initpx)).index(1)
        #start at zeroth planstep always
        initps=0
        return [initturn,initps,initx,0]

    def set_last_prompt(self,prompt):
        self.lastPrompt=prompt

    #called from get_next_action (bayesact.py)
    def get_default_action(self,state):
        if state.get_turn()=="agent":
            (aab,paab)=self.get_default_predicted_action(state)
        else:
            (aab,paab)=self.get_null_action()
            #in any case, get the propositional action (always happens)
            #paab = self.get_prop_action(state)
            
        return (aab,paab)

    def set_behaviour(self,beh):
        self.currBehaviour = beh
        if self.currBehaviour in self.nextBehDict[self.lastPlanStep][1]:
            self.currAwareness=1
        else:
            self.currAwareness=0
        self.currPlanStep = self.getNextPlanStep(self.lastPlanStep,self.currBehaviour)
    #implement a default policy where we look at the client's current awareness
    #and move to the next planstep if is above a threshold
    #if prompted, the threshold will lower
    #called from get_default_action (above)
    #in the real implemented system, this will be getting a measurement of the current planstep from the computer vision system
    #in fact, we get a measure of the client's behaviour, but we can convert this to planstep (e.g. if the behaviour is to do nothing,
    #the planstep stays the same)
    
    def get_prop_action(self,state,rollout=False,samplenum=0):
	awareness = self.currAwareness
        new_awareness=awareness
        print "getting propositonal action for pwid:",awareness,self.lastPrompt,self.currPlanStep,self.deflection_avg

        #save this in case we are running in interactive simulation mode
        #and the user (simulator) puts in something different and then 
        #calls set_behaviour
        self.lastPlanStep=self.currPlanStep
        
        #default next behaviour is to do nothing
        self.currBehaviour = 0
        
        if awareness == 1:
            #the client is aware
            if self.lastPrompt == 0:
                #without a prompt, the client will do the right thing and stay aware if deflection is low
                if NP.random.random() < self.defbnp[min(9,int(round(self.deflection_avg)))]: #> 0.5:
                    self.currBehaviour = self.getNextBehaviour(self.currPlanStep)
                    self.currPlanStep = self.getNextPlanStep(self.currPlanStep,self.currBehaviour)
                #otherwise, loses awareness and stays at same planstep
                else:
                    new_awareness = 0
            else:
                #in a high deflection situation, a prompt will likely confuse the client
                if NP.random.random() > self.defb[min(9,int(round(self.deflection_avg)))]: #0.4:
                    new_awareness = 0
                #will also confuse the client if it is not one that is appropriate
                elif not self.lastPrompt in self.nextBehDict[self.currPlanStep][1]:
                    new_awareness = 0
                #otherwise, a prompt is unlikely to mess things up
                elif NP.random.random() > 0.3:
                    self.currBehaviour = self.getNextBehaviour(self.currPlanStep)
                    self.currPlanStep = self.getNextPlanStep(self.currPlanStep,self.currBehaviour)
                #but if it does, the client may lose awareness, and this is deflection dependent
                elif NP.random.random() > self.defb[min(9,int(round(self.deflection_avg)))]: #0.8:
                    new_awareness = 0
        else:
            #the client is not aware
            if self.lastPrompt == 0:
                #without a prompt
                #very likely that client will do nothing, 
                #but might randomly gain awareness and move on
                if NP.random.random() > 0.99:
                    self.currBehaviour = self.getNextBehaviour(self.currPlanStep)
                    self.currPlanStep = self.getNextPlanStep(self.currPlanStep,self.currBehaviour)
                    new_awareness = 1
            else:
                #with a low-deflection prompt, user will likely move on
                # and gain awareness
                nn=NP.random.random()
                print "nn,defb: ",nn,self.defb[min(9,int(round(self.deflection_avg)))]
                if self.lastPrompt in self.nextBehDict[self.currPlanStep][1] and nn < self.defb[min(9,int(round(self.deflection_avg)))]: #0.1:
                    print "moving on anyways at deflection ",str(self.deflection_avg)," because of random number ",nn
                    self.currBehaviour = self.getNextBehaviourPrompted(self.currPlanStep,self.lastPrompt)   #should be more biased towards lastPrompt 
                    self.currPlanStep = self.getNextPlanStep(self.currPlanStep,self.currBehaviour) 
                    print "behaviour: ",self.currBehaviour," new planstep: ",self.currPlanStep
                    new_awareness = 1
                else:
                    #high deflection prompt will be unlikely to have an effect
                    if NP.random.random() > 0.99:
                        new_awareness = 1
                    
        self.currAwareness = new_awareness
        #this will need to be changed in the real system - we don't actually get a measure of the current planstep, 
        #but rather a measure of the change in planstep, but maybe we can simply convert ...
        return self.currBehaviour

    #this is the only way the pwid finds out what the action of the agent was: its in xobs[1]
    def evalSampleXvar(self,sample,xobs,oldsample=[]):
        self.lastPrompt=xobs[1]
        return 1.0

    #not used
    def reward(self,sample,action=None):
        xreward=0.0
        if self.currPlanStep==self.num_plansteps-1:
            xreward=1.0
        return xreward

