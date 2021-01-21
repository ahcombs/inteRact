"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Assistant Subclass of Agent class
Author: Jesse Hoey  jhoey@cs.uwaterloo.ca   http://www.cs.uwaterloo.ca/~jhoey
September 2013
Use for research purposes only.
Please do not re-distribute without written permission from the author
Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by the Natural Sciences and Engineering Council of Canada (NSERC).
see README for details
----------------------------------------------------------------------------------------------"""

from bayesact import *
import copy

class Assistant(Agent):
    def __init__(self,*args,**kwargs):

        super(Assistant, self).__init__(self,*args,**kwargs)
        #x for the Assistant is a 4-tuple: 
        #the first is the turn (required)
        #the second value is the planstep (0 to N-1) 
        #the third is the binary awareness of the client 0/1
        #the fourth is the last behaviour of the client

        self.obs_noise = kwargs.get("onoise",0.1)

        #dictionary defining the dynamics of the state
        self.nextPsDict = kwargs.get("nextpsd",{0:([1.0],[1]),1:([1.0],[1])})
        self.nextBehDict = kwargs.get("nextbd",{0:{0:([1.0],[0]),1:([1.0],[1])},1:{0:([1.0],[1]),1:([1.0],[1])}})
        self.num_plansteps=len(self.nextPsDict)
        #self.num_behaviours=len(self.nextBehDict)
        self.num_behaviours=kwargs.get("numberbh",5)

        self.reconBehDict = {0:2,1:1,2:2,3:3,4:2,5:2,6:4,7:0}
        #self.reconPsDict = {0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:7}


        #initial awarness distribution
        self.px = [0.3,0.7]

        #self.defb = [1.0,0.99,0.95,0.7,0.5,0.3,0.2,0.1,0.05,0.005]
        #turned these downwards after adding behaviours - seems deflections were not 
        #as high, which may be just because of added uncertainty
        self.defb = [1.0,0.99,0.9,0.3,0.2,0.1,0.05,0.005,0.001,0.0001]

        #self.defbnp = [0.99,0.95,0.7,0.5,0.3,0.2,0.1,0.05,0.005,0.001]
        #self.defbnp = [0.8,0.6,0.5,0.3,0.2,0.1,0.05,0.005,0.002,0.001]
        #downtuned versions
        self.defbnp = [0.8,0.6,0.3,0.2,0.1,0.05,0.005,0.002,0.001,0.0001]


        #observation is only of the planstep (and the turn - [turn,planstep])
        #self.of = (NP.diag(NP.ones(self.num_fplansteps)*(1.0-self.obs_noise-(self.obs_noise/(self.num_plansteps-1)))) +
        #           NP.ones((self.num_plansteps,self.num_plansteps))*(self.obs_noise/(self.num_plansteps-1)))
        #self.of = (NP.diag(NP.ones(self.num_behaviours)*(1.0-self.obs_noise-(self.obs_noise/(self.num_behaviours-1)))) +
        #           NP.ones((self.num_behaviours,self.num_behaviours))*(self.obs_noise/(self.num_behaviours-1)))
        #print self.of
        self.of = [ [ (1.0-self.obs_noise), self.obs_noise/4, self.obs_noise/4, self.obs_noise/4, self.obs_noise/4],
                    [ self.obs_noise,    (1.0-self.obs_noise),     0,                0,                    0 ],
                    [ self.obs_noise,             0,        (1.0-self.obs_noise), 0,                    0       ],
                    [ self.obs_noise,             0,               0,         (1.0-self.obs_noise),     0       ],
                    [ self.obs_noise,             0,               0,               0,      (1.0-self.obs_noise)]]
        print self.of
        
        self.x_avg=[0,0,0,0]
        
        self.lastPrompt = 0

        #propositional (discrete) actions are 0 (do nothing) and a=1...num_plansetps-1 (prompt for planstep a)
        self.numDiscActions=self.num_plansteps   #one action (number 0) is "do nothing" or "None"


    #overridden from Agent because the covariance here is 3D and non-spherical 
    def init_output_covariances(self):

        #self.gamma_value2=self.gamma_value*self.gamma_value

        #precomputed stuff for computing normal pdfs for 3D vectors
        self.theivar=(0.5/float(self.gamma_value[0]*self.gamma_value[0]),
                 0.5/float(self.gamma_value[1]*self.gamma_value[1]),
                 0.5/float(self.gamma_value[2]*self.gamma_value[2]))
                 
        thedet=self.gamma_value[0]*self.gamma_value[1]*self.gamma_value[2]
        self.ldenom=math.log(math.pow((2*NP.pi),1.5)*thedet)



    def print_params(self):
        Agent.print_params(self)
        print "num plansteps: ",self.num_plansteps

        print "observation functin:  ",self.of
        print "ps dynamics: ",self.nextPsDict

    #The only difference between an Assistant and an Agent
    #is dynamics of x and reward
    def initialise_x(self,initx):
        if not initx:
            initpx=self.px
            initturn=State.turnnames.index("agent")  #hard-coded agent start?
        else:
            initpx=initx[1]
            initturn=State.turnnames.index(initx[0])
        #draw a sample from initx over awareness
        initx=list(NP.random.multinomial(1,initpx)).index(1)
        initps=0
        return [initturn,initps,initx,0]


    def is_done(self):
        curr_planstep = self.get_most_likely_planstep()
        return abs(curr_planstep-self.num_plansteps+1)<=0.1


    def get_most_likely_planstep(self):
        ps_belief=[0]*self.num_plansteps
        #ps_votes=[0]*self.num_plansteps
        for s in self.samples:
            #ps_votes[s.x[1]] += 1
            ps_belief[s.x[1]] += s.weight
            
        #print ps_votes
        print ps_belief
        return NP.argmax(ps_belief)


    def get_expected_planstep(self):
        exp_ps=0
        tot_weight=0
        for s in self.samples:
            exp_ps += s.x[1]*s.weight
            tot_weight += s.weight
        return exp_ps/tot_weight
        


    #called from get_next_action (bayesact.py)
    #we could put this in here if we want to assistant to 
    #take a propositional action at each time step
    def get_default_action(self,state):
        if state.get_turn()=="agent":
            (aab,paab)=self.get_default_predicted_action(state)
        else:
            (aab,paab)=self.get_null_action()
            #in any case, get the propositional action (always happens)
            #doesn't happen anymore - 
            #paab = self.get_prop_action(state)
            
        return (aab,paab)

    #implement a default policy where we look at the client's current awareness  (average x[] value)
    #and prompt if this is less than a fixed threshold.
    #the prompt is based on the expected planstep that the client is in
    #should also look at deflection - can we access this here? 
    def get_prop_action(self,state,rollout=False,samplenum=0):

        #for POMCP use this because it will be calling this during rollouts - this is the rollout policy
        #for propositional actions
        if self.use_pomcp:
            if rollout:
                return NP.random.randint(self.numDiscActions)
            else:
                return samplenum%self.numDiscActions #loop through them one by one
            
        else:
            #heuristic policy
            awareness = self.x_avg[2]
            #zero is to do nothing at all
            propositional_action=0
            curr_avg_planstep = round(self.x_avg[1])
            curr_planstep = self.get_most_likely_planstep()
            print "average estimate of planstep: ",curr_avg_planstep," most likely planstep: ",curr_planstep
            expected_planstep = self.get_expected_planstep()
            print "expected planstep: ",expected_planstep

            if awareness < 0.4:
                propositional_action=self.getRecommendedNextBehaviour(curr_planstep)  #used to add 1 here, but that was wrong
            print "using heuristic policy .... awareness: ",awareness," planstep: ",curr_planstep," action : ",propositional_action
            return propositional_action

    #get the next planstep to take according to a plan-graph
    def getNextPlanStep(self,planstep,behaviour):
        newps=self.nextPsDict[planstep][behaviour][1][list(NP.random.multinomial(1,self.nextPsDict[planstep][behaviour][0])).index(1)]
        return newps 
    def getNextBehaviour(self,planstep):
        newbeh=self.nextBehDict[planstep][1][list(NP.random.multinomial(1,self.nextBehDict[planstep][0])).index(1)]
        return newbeh 
        
    #def getRecommendedNextPlanStep(self,planstep):
    #    return min(self.num_plansteps-1,planstep+1)   #in general, could be next step according to some plan-graph
    def getRecommendedNextBehaviour(self,planstep):
        return self.reconBehDict[planstep]

    #aab is the affective part of the action, paab is the propositional part
    def sampleXvar(self,f,tau,state,aab,paab):
        #there is no prompt, the planstep only changes based on awareness
        #and awareness only decreases

        behaviour=state.x[3]

        awareness=state.x[2]
        planstep=state.x[1]
        new_awareness=awareness
        new_planstep=planstep

        new_behaviour = 0  #could be randomized
        #new_behaviour = NP.random.randint(0,self.num_behaviours)  #randomized version


        if state.get_turn()=="agent":
            #agent turn - save action - why?
            self.lastPrompt  = paab
            # we now return without updating because that will happen on the next step when it is client turn
            return [state.invert_turn(),new_planstep,new_awareness,new_behaviour]

        #compute deflection between f and tau
        #x moves towards goalx inversely proportionally to 
        #the deflection
        D=NP.dot(f-tau,f-tau)
        #print "D: ",D," awareness: ",awareness, "planstep: ",planstep, " lastprompt: ",self.lastPrompt,"...new awareness: "
        if awareness == 1:
            #the client is aware
            if self.lastPrompt == 0:
                #without a prompt, the client will do the right thing and stay aware if deflection is low
                if NP.random.random() < self.defbnp[min(9,int(round(D)))]: #> 0.5:
                    new_behaviour = self.getNextBehaviour(planstep)
                    new_planstep = self.getNextPlanStep(planstep,new_behaviour)
                #otherwise, loses awareness
                else:
                    new_awareness = 0
                #without a prompt, the client is likely to do the right thing
                #if NP.random.random() > 0.5:
                #    new_planstep = self.getNextPlanStep(planstep)
                #but may lose awareness randomly, and this increases with deflection  (was a constant 0.5 before)
                #elif NP.random.random() > self.defb[min(9,int(round(self.deflection_avg)))]: #0.5:
                #    new_awareness = 0
            else:
                #print D,self.defb[min(9,int(round(D)))],self.lastPrompt,planstep,awareness,self.nextPsDict[planstep][1]
                #in a high deflection situation, a prompt will likely confuse the client
                if NP.random.random() > self.defb[min(9,int(round(D)))]: 
                    new_awareness = 0
                #otherwise, a prompt for the wrong thing will also mess things up
                elif not self.lastPrompt in self.nextBehDict[planstep][1]:
                    new_awareness = 0
                #otherwise, is unlikely to mess things up unless it is wrong
                elif NP.random.random() > 0.3:
                    new_behaviour = self.getNextBehaviour(planstep)
                    new_planstep = self.getNextPlanStep(planstep,new_behaviour)

                #but if it does, the client may lose awareness, and this is deflection dependent
                elif NP.random.random() > self.defb[min(9,int(round(D)))]: #0.8:
                    new_awareness = 0
        else:
            #the client is not aware
            if self.lastPrompt == 0:
                #without a prompt
                #very likely that client will do nothing, 
                #but might randomly gain awareness and move on
                if NP.random.random() > 0.99:
                    new_behaviour = self.getNextBehaviour(planstep)
                    new_planstep = self.getNextPlanStep(planstep,new_behaviour)
                    new_awareness = 1
            else:
                #with a low-deflection prompt for the next step, user will likely move on
                # and gain awareness
                if NP.random.random() < self.defb[min(9,int(round(D)))] and self.lastPrompt in self.nextBehDict[planstep][1]:
                    new_behaviour = self.getNextBehaviour(planstep)
                    new_planstep = self.getNextPlanStep(planstep,new_behaviour)   #should be more biased towards lastPrompt 
                    new_awareness = 1
                else:
                    #high deflection prompt or incorrect prompt will be unlikely to have an effect
                    if NP.random.random() > 0.99:
                        new_awareness = 1
        return [state.invert_turn(),new_planstep,new_awareness,new_behaviour]
    
    #this must be in the class
    def evalSampleXvar(self,sample,xobs,oldsample=[]):
        #print sample.x[0],sample.x[1],xobs,self.of[sample.x[1]][xobs[1]]
        if sample.x[0]==xobs[0]:
            #print sample.x[3],xobs[1]
            return self.of[sample.x[3]][xobs[1]]
        else:
            return 0.0

    #an observation sample is [turn, planstep]
    def sampleXObservation(self,sample):
        return [sample.x[0],list(NP.random.multinomial(1,self.of[sample.x[1]])).index(1)]


    #this version allows for different variances in each dimension
    def normpdf(self,x, mean, ivar, ldenom):
        num=0.0
        for (xv,iv) in zip(x,ivar):
            num = num-iv*( float(xv) - float(mean) )**2
            
        return num-ldenom


    def reward(self,sample,action=None):
        xreward=0.0
        if sample.x[1] == self.num_plansteps-1:
            xreward = 10.0
        elif sample.x[1] == 1:  #intermediate goal states
            xreward = 0.01
        elif sample.x[1] == 2:  #intermediate goal states
            xreward = 0.01
        elif sample.x[1] == 3:  #intermediate goal states
            xreward = 0.1
        elif sample.x[1] == 4:  #intermediate goal states
            xreward = 0.1
        elif sample.x[1] == 5:  #intermediate goal states
            xreward = 1.0
        elif sample.x[1] == 6:  #intermediate goal states
            xreward = 1.0
        #prompting carries a small cost
        if sample.get_turn()=="agent" and not action[1] == 0:
            xreward = xreward - 0.05

        #larger if out of turn
        #this should really be handled probabilistically I think 
        #so acting out of turn really messes things up 
        if sample.get_turn() == "client" and not action[1] == 0:
            xreward = xreward - 10.0
        return xreward



    #----------------------------------------------------------------------------------
    #POMCP functions
    #----------------------------------------------------------------------------------
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

