"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory 
Agent class with emotions 
Author: Jesse Hoey  jhoey@cs.uwaterloo.ca   http://www.cs.uwaterloo.ca/~jhoey
December 2014
Use for research purposes only.
Please do not re-distribute without written permission from the author
Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by the Natural Sciences and Engineering Council of Canada (NSERC).
use python 2
see README for details
----------------------------------------------------------------------------------------------"""

import os
import math
import sys
import re
import copy
import numpy as NP
import itertools
import time
from pomcp import *
sys.path.append("./gui/")
from cEnum import eTurn

from bayesact import *

emfname=os.path.join(os.path.dirname(__file__), "temotions-male.dat")
effname=os.path.join(os.path.dirname(__file__), "temotions-female.dat")


#open the input file for emotions
#which contains a binary number giving the terms in (fae,fap,faa,ea,ep,ea) where (ea,ep,ea) is the emotions vector
#followed by 3 coefficients
#the first row gives the constant d
#the next three rows give the 3x3 matrix E
#the next three rows give the 3x3 matrix R
#the final row gives the diagonal of the diagonal matrix Q
#for males
fb=file(emfname,"rU")
emot_d=map(lambda(z): float(z),fb.readline().split()[1:])
#print emot_d
emot_E=[]
for i in range(3):
    emot_E.append(map(lambda(z): float(z),fb.readline().split()[1:]))
#actually the transpose
emot_E = NP.array(emot_E).transpose()
#print emot_E
emot_R=[]
for i in range(3):
    emot_R.append(map(lambda(z): float(z),fb.readline().split()[1:]))
emot_R = NP.array(emot_R).transpose()
#print emot_R
#the rest is the Q matrix - 
#this is read generally as follows: 
#Z100XXX means Q_e and multiples fa_e
#Z100010 means the coefficient of e_e multiples fa_p
#Z010XXX means Q_p and multiplies fa_p
#...etc
#we will assume only one fa and one e multiplier per matrix row
emot_Q = {}
for l in ['e','p','a']:
    emot_Q[l] = []
    for i in range(3):
        emot_Q[l].append([0.0,0.0,0.0])
x = fb.readline()
while x!="":
    y=x.split()
    #the f value to use is a 1 in this vector
    fvec=list(y[0])[4:7]
    fvecc=[w[0] for w in filter(lambda (z): z[1]=="1", zip(fvars[0:3],fvec))]
    #fvecc has a string like Fae in it - the third letter gives us which Q matrix this corresponds to
    #the tau dimension we assign to  is given by the position of the 1 in this vector
    evec = list(y[0])[1:4]
    emot_Q[fvecc[0][2]][evec.index('1')] = map(lambda(z): float(z),y[1:])
    x = fb.readline()
fb.close()

for l in ['e','p','a']:
    emot_Q[l] = NP.array(emot_Q[l]).transpose()

#print emot_Q

def computeEmotion(fa,ta):
    normfact = copy.deepcopy(emot_E)
    #possibly don't do the ones that are all zeros here
    for (i,e) in zip([0,1,2],['e','p','a']):
        normfact += NP.diag([fa[i]]*3)*emot_Q[e]
    return NP.dot(NP.linalg.inv(normfact),(ta-NP.dot(emot_R,fa)-emot_d))
        

#now, we should here remove any 'emot_Q' matrices that are all zeros and set them to null
#as we don't want to have to keep re-doing those computations every time



# An EmotionalAgent is an agent that has an additional 3D component for actions, eab, 
# and an additional observation of emotions, 
class EmotionalAgent(Agent):


    def __init__(self,*args,**kwargs):

        #std.dev in the output covariance for emotions
        self.gammae_value = kwargs.get("gammae_value",0.1) 
        
        
        super(EmotionalAgent, self).__init__(self,*args,**kwargs)

        #the file with emotion dictionary in it
        self.fmfname = "fmodifiers.dat"


        self.emotdict = readSentiments(self.fmfname,self.agent_gender)
        self.emotion = {}
        self.emotion["agent"]=NP.zeros(3)
        self.emotion["client"]=NP.zeros(3)


    #really the current emotion should be added into the State ...
    #if its something else, its treated as "agent"
    #returns an EPA vector
    def currentEmotion(self,state,who):
        if who=="client":
            fa = state.f[6:9]
            ta = state.tau[6:9]
        else:
            fa = state.f[0:3]
            ta = state.tau[0:3]
        return computeEmotion(fa,ta)

    #who can be "agent" or "client"
    #if its something else, its treated as "agent"
    #sets self.emotion[who] to be the average EPA vector over the samples,
    #weighted by weights (the expected emotion), and returns it
    def expectedEmotion(self,who):
        if not (who=="agent" or who=="client"):
            who="agent"
        self.emotion[who] = NP.zeros(3)
        sumw=0
        for sample in self.samples:
            theemot = self.currentEmotion(sample,who)
            self.emotion[who] = NP.add(self.emotion[who],sample.weight*theemot)
            sumw += sample.weight
        self.emotion[who] /= sumw
        return self.emotion[who]

    #find closest emotion label is edict [default: self.emotdict] to curremot (EPA vector)
    def findNearestEmotion(self,curremot,edict=None):
        if edict==None:
            edict = self.emotdict
        return findNearestEPAVector(curremot,edict)

    #return the EPA vector for a given emotion label in edict [default: self.emotdict]
    #returns [] if not found
    def lookUpEmotionLabel(self,emot_label,edit=None):
        if edict==None:
            edict = self.emotdict
        if edict.has_key(emot_label):
            return NP.asarray([float(edict[emot_label]["e"]),
                               float(edict[emot_label]["p"]),
                               float(edict[emot_label]["a"])])
        else:
            return []

    #overrides agent version and also initializes for the emotion output distribution
    def init_output_covariances(self):

        super(EmotionalAgent, self).init_output_covariances()

        self.gammae_value2=self.gammae_value*self.gammae_value

        #precomputed stuff for computing normal pdfs for 3D vectors
        theivar=1.0/float(self.gammae_value2)
        thedet=math.pow(float(self.gammae_value),3)
        self.ldenom_e=math.log(math.pow((2*NP.pi),1.5)*thedet)
        self.ivar_e = theivar*0.5


    #additional weight added for how close an observed emotion is to the
    #predicted emotion for the state on turn
    def evalSampleEvar(self,state,turn,observ):
        weight=0.0
        if (not (observ==[] or observ==None)) and turn=="client":
            curr_emot = self.currentEmotion(state,"client")
            dvo=NP.array(curr_emot)-NP.array(observ)
            weight += self.normpdf(dvo,0.0,self.ivar_e,self.ldenom_e)
            #print "comparing ",curr_emot," to ",observ,"weight is ",weight
        else:
            weight=0.0
        return weight




if __name__ == "__main__":
    

    learn_agent=EmotionalAgent()
    #here we are just going to use the emotion calculation functions
    fa=[ 2.16291926,  2.11364315, -0.42280131]
    ta=[.81019615,  0.47730238,  0.10942384] 

    fa = [1.47402725,  0.74727138,  0.53023657]
    ta = [0.84485561, -0.94907817,  0.90339819]


    fa = [1.52991219,  0.8353706,   0.52557234]
    ta = [2.20050925, -1.02906862,  0.93547473]

    fa = [1.60078992,  2.5765918,  -0.48768776] 
    ta = [1.4138979,   1.3309068,   0.28254774]

    fa = [1.9383111,   2.2320633,  -0.89562094]
    ta = [0.21257395,  1.48223805, -0.12987478]

    fa = [1.33251104,  1.40156676,  1.44295723]
    ta = [0.80573881, -0.66668066,  0.61625101]


    fa = [2.34713568,  1.43108062,  2.53255505]
    ta = [1.62898565, -0.17052302,  0.24915186]
    fa=[0.92770291,  1.23166074,  0.95659071]
    ta=[1.3731803,  -0.05628433,  0.26240033]
    curremot = computeEmotion(fa,ta)
    print learn_agent.findNearestEmotion(curremot)
    print curremot
    


