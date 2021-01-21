"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Main classes and functions for POMCP (see below)
Author: Jesse Hoey  jhoey@cs.uwaterloo.ca   http://www.cs.uwaterloo.ca/~jhoey
December 2013
Use for research purposes only.
Please do not re-distribute without written permission from the author
Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by the Natural Sciences and Engineering Council of Canada (NSERC).
use python2.6
see README for details

POMCP algorithm (generic) for problems with continuous state, observation and actions
but with an oracle for the actions: a probability distribution that gives high probability for the optimal action, 
amongst others.  

----------------------------------------------------------------------------------------------------------------------"""
import math
import sys
import re
import numpy as NP
import itertools
import time

#this is also in bayesact.py which is kind of a bummer, but I may want to use the
#two separately, which means I need a third class to put all these basic operations
#also, I need to have a reset here that I don't use in bayesact - very confusing
def drawSamples(samples,N=-1):
    if N==-1:
        N=len(samples)

    new_samples=[]
    weights=map(lambda x: x.weight,samples)
    cumweights=NP.cumsum(weights);
    if cumweights[-1]<1e-200:
        print 100*">","zero weight detected - resetting all weights to 1.0"
        for s in samples:
            new_samples.append(s.get_copy())
            new_samples[-1].weight=1.0
        return new_samples

    cumw=cumweights/cumweights[-1]
    #don't actually need to do this - can just select
    #one random point in the first interval 1..(1/N)
    #and then uniform multiples of 1/N from there
    rsn=NP.random.random_sample((N,))
    rsns=NP.sort(rsn)

    i=0
    for rs in rsns:
        while cumw[i]<rs:
            i+=1
        thesample=samples[i].get_copy()
        thesample.weight=1.0
        new_samples.append(thesample)
    return new_samples

#also in bayesact
def fsampleMean(fsamples):
    if fsamples:
        avgf = fsamples[0]
        for f in fsamples[1:]:
            avgf = avgf+f
        avgf = avgf/len(fsamples)
    
        return avgf
    else:
        return []

#in bayesact
def raw_dist(epa1,epa2):
    dd = NP.array(epa1)-NP.array(epa2)
    return NP.dot(dd,dd)

#kindof in bayesact
def normpdf_old(x, mean, sd):
    size=len(x)
    theivar=1.0/(float(sd)**2)
    thedet=math.pow(theivar,float(size))
    denom=( math.pow((2*NP.pi),float(size)/2) * math.pow(thedet,1.0/2) )
    num=0.0
    for (xv,mv) in zip(x,mean):
        num = num-theivar*0.5*( float(xv) - float(mv) )**2

    return num-math.log(denom)


#---------------------------------------------------------------------------------------------------------------------------
#class to describe the planning tree used by POMCP
#the plan tree could be implemented either with  a fixed number of actions and observations
#or with a fixed resolution.  
#but we need to have a fixed number of actions that we will use as a starting point when doing POMCP
#---------------------------------------------------------------------------------------------------------------------------

class PlanTree(object):
    def __init__(self,numcact,numdact,obsres):
        #number of continuous actions we want to sample
        self.numContActions=numcact

        #number of discrete actions for the discrete factor (must be >0, but can be 1)
        self.numDiscActions=numdact

        #total number of actions
        self.numActions=self.numDiscActions*self.numContActions

        #resolution used to compare observations
        self.obsResolv=obsres

        #root of tree is aplan node
        self.root=PlanNode(0)  #was self.numActions

    def print_val(self,fname=None):
        if fname==None:
            fh=sys.stdout
        else:
            fh = open(fname,"w")
        fh.write("num Continuous actions: "+str(self.numContActions)+"\n")
        fh.write("num Discrete actions: "+str(self.numDiscActions)+"\n")
        fh.write("num Total actions: "+str(self.numActions)+"\n")
        self.root.print_val(0,fh)
        if not fname==None:
            fh.close()

    def print_val_interactive(self):
        write("num Continuous actions: "+str(self.numContActions)+"\n")
        write("num Discrete actions:et "+str(self.numDiscActions)+"\n")
        write("num Total actions: "+str(self.numActions)+"\n")
        self.root.print_val_interactive(0)


    def getBestAction(self):
        return self.root.getBestAction()
    def getWorstAction(self):
        return self.root.getWorstAction()
    def getWorstActionIndex(self):
        return self.root.getWorstActionIndex()


#----------------------------------------------------------------------------------------------------------------------------
#class for a node in the tree
#----------------------------------------------------------------------------------------------------------------------------

class PlanNode(object):
    def __init__(self,numa):
        self.children=[]
        self.B=[]
        self.N=0
        self.Nh=[]
        self.actionSet=[]
        self.observSet=[]
        self.observSetData=[]
        self.children=[]
        self.Vh=[]
        self.numActions=numa

    #the actions in the action set are (affective,propositional)
    #where affective is a 3D vector and propositional is an integer index
    def addAction(self,a,iVal):
        self.actionSet.append(a)
        self.children.append([])
        self.observSet.append([])
        self.observSetData.append([])
        self.Vh.append(iVal)  #should be r_hi (highest return from sample rollouts) computed automatically
        self.num_init=10
        self.Nh.append(self.num_init)  #unclear what this should be - possibly the number of rollouts performed? 
        self.numActions += 1
        self.N=self.num_init*self.numActions


    def replaceAction(self,index_replace,a):
        self.actionSet[index_replace]=a
        self.children[index_replace]=[]
        self.observSet[index_replace]=[]
        self.observSetData[index_replace]=[]
        self.Vh[index_replace]=40  #should be r_hi (highest return from sample rollouts) computed automatically
        self.Nh[index_replace]=10

    def getBestAction(self):
        (bestv,besta)=max( (v, i) for i, v in enumerate(self.Vh) )    #this may throw a ValueError() max: arg is an empty sequence
        return (bestv,self.actionSet[besta])

    def getWorstAction(self):
        (worstv,worsta)=self.getWorstActionIndex()
        return (worstv,self.actionSet[worsta])

    def getWorstActionIndex(self):
        (worstv,worsta)=min( (v, i) for i, v in enumerate(self.Vh) )
        return (worstv,worsta)

    def print_aset(self,tab,fh=sys.stdout):
        index=0
        for a in self.actionSet:
            #this version pretty-prints the floats but doesn't work in the case of stochastic actions
            #fh.write(str(tab)+tab*" "+str(index)+": "+str(["%0.2f" % i for i in a[0]])+"..."+str(["%s" % i for i in a[1:]])+"\n")
            fh.write(str(tab)+tab*" "+str(index)+": ")
            print a
            index=index+1

    def print_oset(self,tab,fh=sys.stdout):
        aind=0
        for a in self.observSet:
            fh.write(str(tab)+tab*" "+"for action "+str(aind)+"\n")
            jndex=0
            for o in a:
                #o[0] is the continuous observation, and o[1:] are the discrete observations
                #should be generalised
                fh.write(str(tab)+tab*" "+str(jndex)+": "+str(["%0.2f" % i for i in o[0]])+"..."+str(["%s" % i for i in o[1:]])+"\n")
                fh.write("number of observations at this oset: "+str(len(self.observSetData[aind][jndex]))+"\n")
                #for xx in self.observSetData[aind][jndex]:
                #    print (xx)
                jndex=jndex+1
            aind += 1
            fh.write(str(tab)+tab*" "+"mean: "+str(fsampleMean(map(lambda x: NP.array(map(lambda y: float(y),x[0])),a)))+"\n")

    def print_level_val(self,tab=0,fh=sys.stdout):
        fh.write(str(tab)+tab*" "+"Vh: "+str(self.Vh)+"\n")
        fh.write(str(tab)+tab*" "+"Nh: "+str(self.Nh)+"\n")
        fh.write(str(tab)+tab*" "+"actions: \n")
        self.print_aset(tab,fh)
        fh.write(str(tab)+tab*" "+"observs: \n")
        self.print_oset(tab,fh)
        #fh.write(str(tab)+tab*" ","beliefs: \n")
        #for b in self.B:
        #    fh.write(str(tab)+tab*" ",b.print_val()

    def print_val(self,tab=0,fh=sys.stdout):
        #don't print if the node has no children
        if self.children==[]:
            return
        if self.children[0]==[]:
            return
        self.print_level_val(tab,fh)
        fh.write(str(tab)+tab*" "+25*"-"+"children"+25*"-"+"\n")
        ci=0
        for c in  self.children:
            cii=0
            for child in c:
                fh.write(str(tab)+tab*" "+"child for action "+str(ci)+" observation "+str(cii)+"\n")
                child.print_val(tab+1,fh)
                cii += 1
            ci += 1
        fh.write(str(tab)+tab*" "+65*"-"+"\n")

    def print_val_interactive(self,tab=0):
        
        fh=sys.stdout
        goback=0
        while goback>=0:
            self.print_level_val(tab,fh)
            theact = int(raw_input("action to follow (0-"+str(len(self.children)-1)+") (negative -n goes back by n levels): "))
            if theact<0:
                return theact+1

            if theact >= len(self.observSet) or len(self.observSet[theact])==0:
                continue

            theobs = int(raw_input("observation to follow (0-"+str(len(self.observSet[theact])-1)+"): "))

            if theobs >= len(self.children[theact]):
                continue
                
            goback = self.children[theact][theobs].print_val_interactive(tab+1)
            if (goback+1)<=0:
                return goback+1
        print "did not return?"
        print goback
        
        
        
        
        

class POMCP(object):
    
    #initialise parameters of POMCP algorithm
    #cval is the confidence bound weight in UCB
    #numact is the number of actions to sample at each level
    #obsres is the resolution of observations (to discretise them)
    #timeout is the timeout in seconds (statically set for now)
    def __init__(self,*args,**kwargs):
        self.c_val=kwargs.get("cval",1.0)
        self.numContActions = kwargs.get("numcact",1)
        self.numDiscActions = kwargs.get("numdact",1)
        self.numActions = self.numContActions*self.numDiscActions
        #number of actions to add on each call to simulate - default 1, but could be self.numActions
        #if we don't want to do things incrementally
        self.numAddActions = kwargs.get("numaddact",1)
        self.obsResolv = kwargs.get("obsres",1.0)
        self.actResolv = kwargs.get("actres",0.1)
        self.pomcp_timeout = kwargs.get("timeout",1.0)
        #set up the tree to start with 
        self.ucTree=PlanTree(self.numContActions,self.numDiscActions,self.obsResolv)
        self.epsilon = kwargs.get("epsilon",0.1)

        self.verbose = kwargs.get("verbose",False)
        self.stateBufferSize=100

        


    def POMCP_interactiveExplorePlanTree(self):
        self.ucTree.root.print_val_interactive()
        

    def printParams(self):
        print("self.c_val",self.c_val)
        print("self.numContActions",self.numContActions)
        print("self.numDiscActions",self.numDiscActions)
        print("self.numActions",self.numActions)
        print("self.obsResolv",self.obsResolv)
        print("self.actResolv",self.actResolv)
        print("self.pomcp_timeout",self.pomcp_timeout)
        print("self.stateBufferSize",self.stateBufferSize)


    def POMCP_printLevelVal(self):
        self.ucTree.root.print_level_val()


    def POMCP_pruneTree(self,action_taken,observation_made,blackBox):

        oldroot = self.ucTree.root

        if self.verbose:
            print "pruning tree with action taken: ",action_taken," and observation made: ",observation_made
        
        besta=-1
        bestd=-1
        besto=-1

        firstone=True
        #find closest action to the one taken
        for aindex in range(len(oldroot.actionSet)):
            #only look at the ones that match - if adist<0 there is no match
            adist=blackBox.actionDist(oldroot.actionSet[aindex],action_taken)
            #print adist,oldroot.actionSet[aindex],action_taken
            if (firstone and adist>=0)  or ((not firstone) and adist>=0 and adist < bestd):
                firstone=False
                besta=aindex
                bestd=adist

        
        if self.verbose:
            print "pruning tree .."
            print "best action: ",besta, "distance ",bestd

        #may happen at start if we prune the tree before growing it
        #if so, we simply return negative values and the tree is not pruned, but this
        #is never flagged up anywhere.  
        if besta<0:
            if self.verbose:
                print 100*"!","error - action not found in POMCP tree, something is likely wrong somewhere ..."
            return  (besta,besto,bestd)

        #find closest observation to one obtained
        firstone=True
        bestd=-1
        for oindex in range(len(oldroot.children[besta])):
            #only look at the ones that match - if adist<0 there is no match
            #print oldroot.observSet[besta][oindex],observation_made
            odist=blackBox.observationDist(oldroot.observSet[besta][oindex],observation_made)
            if (firstone and odist>=0)  or ((not firstone) and odist>=0 and odist < bestd):
                firstone=False
                besto=oindex
                bestd=odist

        if self.verbose:
            print "best observation: ",besto," distance ",bestd
            print "number of children searched ",len(oldroot.children[besta])

        #means this observation was not found - keep the tree the same? 
        #no, we should reset, but not sure how to do this
        if besto<0:
            if self.verbose:
                print 100*"!","error - observation not found in POMCP tree, something is likely wrong somewhere"
            return (besta,besto,bestd)

        self.ucTree.root = oldroot.children[besta][besto]

        return (besta,besto,bestd)


    #initial evaluation of the the POMCP problem to get Rhi and Rlo and Cval
    #does a set of rollouts and finds highest and lowest reward
    def POMCP_eval(self,samples,numrollouts,blackBox):
        #draw an initial set of unweighted samples
        thesamples=drawSamples(samples,numrollouts)
        firstone=True
        for sample in thesamples:
            gotRew=self.rollout(sample,0,blackBox)
            if firstone or gotRew>bestRew:
                bestRew=gotRew
            if firstone or gotRew<worstRew:
                worstRew=gotRew
                firstone=False
        self.rhi=bestRew
        self.rlo=worstRew
        self.c_val=self.rhi-self.rlo
        if self.verbose:
            print "rhi: ",self.rhi," rlo: ",self.rlo," new cval: ",self.c_val
        
    #get the possible actions at the root 
    #used for monitoring only
    def getPossibleNextActions(self):
        return self.ucTree.root.print_aset(0)


    #POMCP search is called with a belief state represented
    #as a set of weighted samples
    #returns a tuple of (value,bestaction)
    def POMCP_search(self,samples,blackBox,timeout=None):
        if timeout == None:
            timeout = self.pomcp_timeout

        #draw an initial set of unweighted samples
        thesamples=drawSamples(samples,self.stateBufferSize)

        start = time.clock()
        
        sampleindex=0
        #temporarily change to num iterations for repeatability during debugging or otherwise
        numiters=0
        #while numiters < timeout:
        while (time.clock()-start) < timeout:
            #print "sample :",thesamples[sampleindex].print_val()
            self.POMCP_simulate(self.ucTree.root,thesamples[sampleindex],0,blackBox)
            
            sampleindex = sampleindex+1
            if sampleindex >= self.stateBufferSize:
                #replace with a new set
                thesamples=drawSamples(samples,self.stateBufferSize)
                sampleindex=0
            numiters = numiters + 1

        if self.verbose:
            print "searched ",numiters," iterations of POMCP_search"
        xx = self.ucTree.getBestAction()
        return self.ucTree.getBestAction()


    def POMCP_getBestAction(doucb=False):
        #pick best action to take with the UCB formula
        for a in range(self.ucTree.numActions):
            bval = ucNode.Vh[a]
            if doucb:
                bval = bval + self.c_val*math.sqrt(math.log(ucNode.N)/ucNode.Nh[a])
            if a==0 or bval>maxbval:
                maxbval=bval
                besta = a
        return (besta,maxbval)

        
    #blackBox argument is an object with a propagateSample(state,action) method that can be called and returns a sample
    def POMCP_simulate(self,ucNode,state,depth,blackBox):
        
        
        if math.pow(blackBox.discount_factor,depth) < self.epsilon:
            return 0

        if ucNode.numActions<self.numActions:   #keep adding until we reach max number
            
            gotNewAction=False
            #draw a set of samples of actions as the new set
            for aindex in range(self.numAddActions):
                #get the action to take - returns by the oracle which can just spit out
                #random actions, or can go through a sequence by using the third argument
                theaction=blackBox.oracle(state,False,ucNode.numActions)
                if ucNode.numActions<self.numActions and not blackBox.hasAction(ucNode.actionSet,theaction,self.actResolv):
                    ucNode.addAction(theaction,self.rhi)
                    gotNewAction=True

            if gotNewAction:
                rolloutval = self.rollout(state,depth,blackBox)
                #print "rolloutval: ",rolloutval
                return rolloutval
            
        
        #pick best action to take with the UCB formula
        #any new ones that were added will be explored more
        #print "ucNode.numActions: ",ucNode.numActions
        #print "ucNode.Vh: ",ucNode.Vh
        #print "ucNode.N: ",ucNode.N
        #print "ucNode.Nh: ",ucNode.Nh
        for a in range(ucNode.numActions):
            bval = ucNode.Vh[a]+self.c_val*math.sqrt(math.log(ucNode.N)/ucNode.Nh[a])
            if a==0 or bval>maxbval:
                maxbval=bval
                besta = a

        #besta=1
        #print 100*"%"
        #print "best action: ",besta

        #simulate next state given that action 
        (newsample, newobs, newreward) = blackBox.propagateSample(state,ucNode.actionSet[besta])

            
        #newobs is a tuple with continuous then discrete values
        
        #print 100*"~"
        #print ucNode.actionSet[besta]
        #print state.x,"--->",newsample.x
        #print newxobs

        #may want to compare ucNode.B and state here.  It may be necessary to draw some samples from 
        #a different distribution, possibly ucNode.B itself, or possibly the uniform distribution, etc.
        ucNode.B.append(state)



        #print "reward: ",newreward
        #find closest observation to newobs in current list of observations that have been sampled at this node-action 
        #if this closest observation is still further away than obsResolv, we add it as new
        obsExists=-1
        #print len(ucNode.children[besta])
        #print 100*"-"
        #print besta
        #print ucNode.observSet[besta]
        #print newobs
        (oindex,bestdist)=blackBox.bestObservationMatch(newobs,ucNode.observSet[besta],ucNode.observSetData[besta])

        #print 100*"^"
        #print oindex
        #print bestdist
        #print newobs
        #print ucNode.observSet[besta]
        #print 100*"^"

        if oindex>=0 and bestdist < self.ucTree.obsResolv:
            #here, we should average the observations so far at this node and make the mean be the observation
            #print "adding to existing"
            ucNode.observSetData[besta][oindex].append(newobs)
            ucNode.observSet[besta][oindex]=blackBox.getMeanObs(ucNode.observSetData[besta][oindex])
            #print "mean values:  ", ucNode.observSet[besta][oindex]
            obsExists=oindex
        #if this observation is new
        else:
            #print "new ",100*"!"
            #print 100*"^"
            #print oindex
            #print bestdist
            ##print newobs
            #print ucNode.observSet[besta]
            #print 100*"^"


            ucNode.observSet[besta].append(newobs)
            ucNode.observSetData[besta].append([])
            ucNode.observSetData[besta][-1].append(newobs)
            ucNode.children[besta].append(PlanNode(0)) #was: ucNode.numActions))
            #obsExists stays as -1 as it is now the last element

        thereward = newreward+blackBox.discount_factor*self.POMCP_simulate(ucNode.children[besta][obsExists],newsample,depth+1,blackBox)
        ucNode.N += 1
        ucNode.Nh[besta] += 1
        ucNode.Vh[besta] += (thereward-ucNode.Vh[besta])/ucNode.Nh[besta]
        return thereward
                

    def rollout(self,state,depth,blackBox):
        
        if math.pow(blackBox.discount_factor,depth) < self.epsilon:
            return 0
        
        #rollout policy selects near the optimal myopic action to take
        a=blackBox.oracle(state,True)

        #simulate next state given that action 
        (newsample, newobs, newreward) = blackBox.propagateSample(state,a)
    
        return newreward+blackBox.discount_factor*self.rollout(newsample,depth+1,blackBox)

