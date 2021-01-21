"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Main classes and functions
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

import os
import math
import sys
import re
import copy
import numpy as NP
import itertools
import time
import random
from pomcp import *
sys.path.append("./gui/")
from cEnum import eTurn

#from hfun import h_fun as h_fun_c

#---------------------------------------------------------------------------------------------------------------------------
#filenames for the ACT databases
#---------------------------------------------------------------------------------------------------------------------------
ofname="bayesact.spudd"
#file with the transient equations
temfname=os.path.join(os.path.dirname(__file__), "tdynamics-male.dat")
teffname=os.path.join(os.path.dirname(__file__), "tdynamics-female.dat")
#wild test
#teffname=os.path.join(os.path.dirname(__file__), "tdynamics-test.dat")
#tests where transients are equal to fundamentals
#temfname="tdynamics-test.dat"
#teffname="tdynamics-test.dat"

sentiment_levels=2
smin=-4.3
smax=4.3
sincr=(smax-smin)/sentiment_levels



#for discretising the sentiment range - not really used anymore
sentiment_indices=xrange(sentiment_levels+1)
sentiment_bins=[x*sincr+smin for x in list(xrange(sentiment_levels+1))]




#---------------------------------------------------------------------------------------------------------------------------
#Some predefined constants
#---------------------------------------------------------------------------------------------------------------------------

#whether the fundamental identities are statically defined or not
f_static=False
#whole model is observable  - this is implied by x and f being observable
f_observable=False


#normally this is ["e","p","a"]
epa=["e","p","a"]
#this is usually  ["a","b","c"]
sentvals=["a","b","c"]


#create names for fundamentals and transients
fvars=[]
tvars=[]
#switched these around feeb 15th used to be e first then s
for s in sentvals:
    for e in epa:
        tvars.append("T"+s+e)
        if not f_static or not (s=="a" or s=="c"):
            fvars.append("F"+s+e)



#---------------------------------------------------------------------------------------------------------------------------
#Builds some global constants describing dynamics (this is the matrix "M")
#---------------------------------------------------------------------------------------------------------------------------

#open the input file for dynamics of transients
#which contains a binary number giving the terms in tvars to be used
#followed by the 9-D multiplication matrix M row
# the tvars are ordered as:
# Ae, Ap, Aa, Be, Bp, Ba, Oe, Op, Oa,
#the result is e.g. tdyn-male which is an array of arrays, each of which
#contains a list of tvars terms followed by the matrix row
#for males
tdyn_male=[]
fb=file(temfname,"rU")
x=fb.readline()
while x!="":
    y=x.split()
    tvec=list(y[0])[1:]
    tvecc=[w[0] for w in filter(lambda (z): z[1]=="1", zip(tvars,tvec))]
    M=map(lambda(z): float(z),y[1:])
    tdyn_male.append([tvecc,M])
    x=fb.readline()
fb.close()
#for females
tdyn_female=[]
fb=file(teffname,"rU")
x=fb.readline()
while x!="":
    y=x.split()
    tvec=list(y[0])[1:]
    tvecc=[w[0] for w in filter(lambda (z): z[1]=="1", zip(tvars,tvec))]
    M=map(lambda(z): float(z),y[1:])
    tdyn_female.append([tvecc,M])
    x=fb.readline()
fb.close()


#---------------------------------------------------------------------------------------------------------------------------
#statically defined functions of general usage
#---------------------------------------------------------------------------------------------------------------------------
#initialises ids
#0: knows own id, not client id   (so input argument client_id is ignored)
#1: knows own id, knows client id is one of num_confusers possibilities
#2: knows own id, knows client id
#3: doesn't know own id, doesn't know client id
#4: has multiple self-ids, but either an unknown or a single client id
#5: has mulitple self-ids and multiple client-ids (all specified)
def init_id(knowledge,agent_id,client_id=[],mean_id=[],num_confusers=0,aidprop=[]):
    
    if knowledge==2 and not client_id == []:
        tau_init=NP.concatenate((agent_id,NP.zeros((3,1)),client_id)).transpose()
        prop_init=[1.0]
        beta_client_init=0.1  #0.25 
        beta_agent_init=0.1 #0.25
    elif knowledge==1:
        for i in range(num_confusers+1):
            tmp_id = NP.dot(NP.random.random([3,1]),8.6)-4.3
            tau_init.append(NP.concatenate((agent_id,NP.zeros((3,1)),tmp_id)))
        ci=NP.random.randint(num_confusers+1)
        tau_init[ci]=NP.concatenate((agent_id,NP.zeros((3,1)),client_id))
        prop_init=NP.dot(NP.ones((num_confusers+1,1)),1.0/(num_confusers+1)).transpose()
        beta_client_init=0.5
        beta_agent_init=0.01
    elif knowledge==3:
        if mean_id==[]:
            tau_init=NP.concatenate((NP.zeros((3,1)),NP.zeros((3,1)),NP.zeros((3,1)))).transpose()
        else:
            tau_init=NP.concatenate((NP.asarray([mean_id]).transpose(),NP.zeros((3,1)),NP.asarray([mean_id]).transpose())).transpose()
        prop_init=[1.0]
        beta_client_init=2.0
        beta_agent_init=2.0
    elif knowledge==4:
        tau_init=[]
        if client_id==[]:
            #agent does not know client id, but has multiple self ids
            for aid in agent_id:
                if mean_id==[]:
                    tau_init.append(NP.concatenate((aid,NP.zeros((3,1)),NP.zeros((3,1)))).transpose()[0])
                else:
                    tau_init.append(NP.concatenate((aid,NP.zeros((3,1)),NP.asarray([mean_id]).transpose())).transpose()[0])
            prop_init=aidprop
            beta_client_init=2.0
            beta_agent_init=0.1  #was 0.01
        else:
            #agent knows client id, but has multiple self ids
            tau_init=[]
            for aid in agent_id:
                tau_init.append(NP.concatenate((aid,NP.zeros((3,1)),client_id)).transpose()[0])
            prop_init=aidprop
            beta_client_init=0.5
            beta_agent_init=0.01
    elif knowledge == 5:
        tau_init = []
        num_ids = 0
        for aid in agent_id:
            for cid in client_id:
                tau_init.append(NP.concatenate((aid,NP.zeros((3,1)),cid)).transpose()[0])
                num_ids += 1
        #all equally probable
        prop_init=NP.dot(NP.ones((num_ids,1)),1.0/(num_ids)).transpose()
        prop_init = prop_init[0]
        beta_client_init=0.1
        beta_agent_init=0.1
    else:   #knowledge is 0
        if mean_id==[]:
            tau_init=NP.concatenate((agent_id,NP.zeros((3,1)),NP.zeros((3,1)))).transpose()
        else:
            tau_init=NP.concatenate((agent_id,NP.zeros((3,1)),NP.asarray([mean_id]).transpose())).transpose()
        prop_init=[1.0]
        beta_client_init=2.0
        beta_agent_init=0.01
    return (tau_init,prop_init,beta_client_init,beta_agent_init)



def make_ids_covs(fifname,ids,agender):
    for theid in ids:
        if type(theid[1])==type("dummy"):   # its a label so we need to look it up
            if len(theid)<4:
                gender=agender
            else:
                gender=theid[3]
            theid[1]=getIdentity(fifname,theid[1],gender)
        #why is this here????????? - I'm squaring the id again which is bizarre
        #something to do with bayesactself, so I am removing it for now
        #actually this is necessary as theid[2] is a single number - the variance, 
        #so we square it and make it diagonal.  Why did I comment this out? 
        theid[2] = NP.eye(3)*(theid[2]**2)
    return ids



#reads in sentiments for gender from fbfname and returns a dictionary
def readSentiments(fbfname,gender):
    #open the input file for fundamentals of behaviours or identities
    addto_agent=0
    if gender=="female":
        addto_agent=3
    fsentiments_agent={}
    fb=file(fbfname,"rU")
    for x in set(fb):
        y=[re.sub(r"\s+","_",i.strip()) for i in x.split(",")]
        #sentiment_bins include the end-points -4.3 and 4.3 so we strip them off.
        z=["fb"+str(j) for j in NP.digitize(y[1:-1],sentiment_bins[1:-1])]
        #1,2,3 areEPA for males, 4,5,6 are EPA for females
        #for males
        fsentiments_agent[y[0]]={"elabel":z[0+addto_agent],"e":y[1+addto_agent],
                                 "plabel":z[1+addto_agent],"p":y[2+addto_agent],
                                 "alabel":z[2+addto_agent],"a":y[3+addto_agent]}
        if len(y) > 7:
            institution_list  = map(lambda i: bool(int(i)),list(re.sub(r"_","",y[7])))
            fsentiments_agent[y[0]]["institutions"]=institution_list

    fb.close()
    return fsentiments_agent

#returns the mean and covariance of all identities in the dictionary for gender
def getIdentityStats(fifname,gender):
    iddic=readSentiments(fifname,gender)
    meanid = NP.zeros((1,3)).flatten(1)
    covid = NP.zeros((3,3))
    N=0
    for theid in iddic:
        thevec=NP.asarray([float(iddic[theid]["e"]),float(iddic[theid]["p"]),float(iddic[theid]["a"])])
        meanid = NP.add(meanid,thevec)
        N=N+1
    meanid = meanid/N
    for theid in iddic:
        thevec=NP.add(NP.asarray([float(iddic[theid]["e"]),float(iddic[theid]["p"]),float(iddic[theid]["a"])]),NP.dot(-1.0,meanid))
        covid = NP.add(covid,NP.dot(NP.transpose([thevec]),[thevec]))
    covid = covid/(N-1)
    return (meanid,covid)

#prints all pairs of identities that are within edist of each other (euclidean distance)
def printNearbyIdentities(fifname,gender,edist):
    iddic=readSentiments(fifname,gender)
    listofids=[]
    for theid in iddic:
        thevec=NP.asarray([float(iddic[theid]["e"]),float(iddic[theid]["p"]),float(iddic[theid]["a"])])
        for theotherid in iddic:
            if not theid == theotherid:
                thedist=math.sqrt(fdist(thevec,iddic[theotherid]))
                if thedist < edist:
                    listofids.append((theid,theotherid,thedist))
    return listofids

#gets the EPA value for identity ident with gender
def getIdentity(fifname,theid,gender):
    iddic=readSentiments(fifname,gender)
    if iddic.has_key(theid):
        return NP.asarray([float(iddic[theid]["e"]),float(iddic[theid]["p"]),float(iddic[theid]["a"])])
    else:
        ## strip off spaces and replace interior spaces with underscores in csae this works
        tryagain = re.sub(r"\s+","_",theid.strip())
        if iddic.has_key(tryagain):
            theid = tryagain
            return NP.asarray([float(iddic[theid]["e"]),float(iddic[theid]["p"]),float(iddic[theid]["a"])])
        else:
            return []

def invert_turn(turn):
    if turn=="agent":
        return "client"
    elif turn=="client":
        return "agent"
    else:
        return turn


#true if mrow[0] has a variable that starts with v in it
def has_variable(mrow,v):
    return reduce(lambda xx,yy: xx or yy.startswith(v),mrow,False)


def swapACVar(Tvar):
    if Tvar[1]=="a":
        return Tvar[0]+"c"+Tvar[2]
    elif Tvar[1]=="c":
        return Tvar[0]+"a"+Tvar[2]
    else:
        return Tvar

def swapACVars(arrayTvars):
    return map(lambda x: swapACVar(x), arrayTvars)

#also make swapped versions of these with all input and output a and c swapped
#still to be done -----
def swapAC(dynVar):
    dd=[]
    for d in dynVar:
        ddd=[]
        ddd.append(swapACVars(d[0]))
        # this is a bummer to have this fixed, but much more difficult than otherwise
        # we know the order is : Ae, Ap, Aa, Be, Bp, Ba, Oe, Op, Oa,
        ddd.append(NP.concatenate((d[1][6:9],d[1][3:6],d[1][0:3])))

        dd.append(ddd)
    return dd


#Tvar is the name of a variable in tdyn
#tvars is the array of transient variable names
#fvars is the array of fundamental variable names
#tdyn[0] is an array of arrays of transient names, each one giving the transients involved in that factor in "t"
#tdyn[1] is an array of coefficients (the matrix M)
#tau is an array of tau values for the current step  (in the same order as tvars)
#fp is an array of fundamental values for the next time step (in the same order as tvars)
#sampleTvar("Tae",tvars,tdyn,[array of Tau values],[array of f_primed values]
#although this is called "sample"Tvar, it is actually a deterministic function!
def sampleTvar(Tvar,tvars,GM,tau,fp,turn):
    sum=0
    for (tvarnames,ovecval) in GM[turn][Tvar]:
        term=1
        for tvarname in tvarnames:
            if tvarname[1]=="b":
                fvarname="F"+tvarname[1:] #its taken from the primed fundamentals variable with the same e and abo values
                term *= fp[tvars.index(tvarname)]  #could be fp[fvarname] - so make fp a dictionary over variable names?
            else:
                term *= tau[tvars.index(tvarname)] #its taken from the transients variable from the previous step
                                                   #could be tau[tvarname] if tau is a dictionary
        term *= ovecval  #was ... why? : M[tvars.index(tvarname)]
        sum+=term
    return sum




#this is a version of instHCFromTau that only uses map etc
#but it seems to be slower than the h_fun version below, so is not used.
def instHCfromTauNew(tvars,GG,MM,tau,turn):

    #tmpH=NP.dot(map(lambda x: NP.prod(map(lambda y: tau[tvars.index(y)],x)),GG[turn]['Tbe']),MM[turn]['Tbe'])
    tmpH=NP.dot([NP.prod([tau[tvars.index(y)] for y in x]) for x in GG[turn]['Tbe']],MM[turn]['Tbe'])

    for j in ["Tbp","Tba"]:
        tmpH=NP.vstack([tmpH,NP.dot([NP.prod([tau[tvars.index(y)] for y in x]) for x in GG[turn][j]],MM[turn][j])])

    tmpC=NP.array([NP.dot([NP.prod([tau[tvars.index(y)] for y in x]) for x in GG[turn]['nob']],MM[turn]['nob'])]).transpose()

    #returns the instance of H and C needed from tau
    return (tmpH,tmpC)


#this function builds the instances of \mathscr{H} and \mathsrc{C} from tau for a particular turn
#essentially, this constructs \mathscr{G} but without any "b" terms (no behaviours - replacing any elements of \mathscr{G}
#that have only behaviours with 1), replicates it row-wise 9 times to give a 9x29 matrix, and mutliplies
#this point-wise by the transpose of the transition matrix, M^T (another 9x29 matrix).
#It then divides the resulting matrix into four blocks corresponding to the elements of the original \mathsrc{G}
#that contain no, 'e', 'p' and 'a' behaviours (none,Tbe,Tbp,Tba).
#It then sums each of these blocks to give a 9x4 matrix, the first column of which is \mathsrc{C} and
#the last three columns of which are \mathsrc{H}
#inputs are
# - tau (the current transient sentiment vector)
# - H the rows of the transition matrix M that correspond to values of \mathscr{G} with Tb<i> values i=e,p, or a
#     H[turn]['Tbe'] gives those rows that correspond to values of \mathsrc{G} that contain 'Tbe'
# - C the rows of the transition matrix M that correspond to values of \mathscr{G} with no Tb values at all
#H[turn]['Tbi'] and C[turn] are both lists of tuples, each tuple giving: first, a list of strings like 'Tbi'
#giving the set of transients from \mathsrc{G} that should be multiplied together; second, the corresponding row from M
def instHCfromTau(tvars,H,C,tau,turn):
    tmpH=h_fun(H[turn]["Tbe"],tvars,tau)
    for j in ["Tbp","Tba"]:
        tmpH= NP.vstack([tmpH,h_fun(H[turn][j],tvars,tau)])
    tmpC=NP.array([h_fun(C[turn],tvars,tau)]).transpose()
    #returns the instance of H and C needed from tau
    return (tmpH,tmpC)


#consumes a matrix h which is like tdyn -  has h[i][0] giving the elements of of T that are to be used
#(a list of strings of the form 'Tij' where i is a,b,c and j is e,p,a
#and h[i][1] giving the coefficients of M for that element
def h_fun(h,tvars,t):
    #vector of zeros of the same size as h[1]
    #sum=NP.zeros_like(h[0][1])
    sum=[0]*9
    for hh in h:
        #only do the non-behaviour transient values
        #mult = 1
        #for v in filter(lambda xx: not xx[1]=="b",hh[0]):    #hacked in indices here - yuck
            #mult *= t[tvars.index(v)]   # this was when hh[0] contained strings, now it contains indices
        #for v in hh[0]:
        #    mult *= t[v]
        if hh[0]:
            mult=t[hh[0][0]]
            for v in hh[0][1:]:
                mult *= t[v]
        else:
            mult=1
        #these are all different ways of doing the same thing
        #mult=NP.prod([t[v] for v in hh[0]])
        #sum = NP.add(sum,NP.multiply(mult,hh[1])) # pointwise multiply so return the whole row h[1]  multiplied by all the t[v]'s
        #sum = [sum[i]+mult*hh[1][i] for i in range(9)]
        #this, the least fancy method seems to work fastest!
	for i in range(9):
	    sum[i] = sum[i] + mult*hh[1][i]
    return sum


def sampleTvars(tvars,instH,instC,fp):
    term = NP.add(NP.array([NP.dot(fp[3:6],instH)]).transpose(),instC)
    return term.flatten(1)



#extracts the b variables from fvars
#flag =False : gets the "not-b" variables from fvars
def getbvars(fvars,vars,flag=True):
    if flag:
        return map(lambda x: x[1], filter(lambda x: x[0][1]=="b", zip(fvars,vars)))
    else:
        return map(lambda x: x[1], filter(lambda x: not x[0][1]=="b", zip(fvars,vars)))


class FVarSampler:
    def _computeWeight(self):
        return 0.0;

    def _computeMuSig(self, f, isiga, isigf):
        zmat= NP.zeros((3,9))
        I = NP.eye(9)
        k0 = NP.vstack([zmat, self.h, zmat])
        k0 = k0.transpose()
        K = I - k0
    
        #including beta, full version using a draw from a multivariate_normal
        #this will be slower I guess, but I can't find a way to split these into two
        #one way might be to sample without the beta term, then weight using the difference
        # from the previous fundamental
        ka=NP.dot(K.transpose(),isiga)
    
        #this is K^{-1} \Sig K^{-1}  or (K\Sig^{-1} K)^{-1}
        #this is actually (K\Sig^{-1} K)  - not sure what the previous comment line was supposd to mean - delete?
        mean_prediction = NP.dot(ka, K)
        val = mean_prediction + isigf
        sig_n = NP.linalg.inv(val) # TODO Optimize

        #mean value
        mu_n = NP.dot(sig_n,(NP.dot(ka,self.c)+NP.dot(isigf,NP.array([f]).transpose())))

        #f_s = NP.random.multivariate_normal(NP.asarray(mu_n).flatten(1),sig_n).transpose()
        #TODO: Verify correctness (use above line as reference)
        mu_n = NP.asarray(mu_n).flatten(1)

        return (mu_n, sig_n)
        

    def __init__(self, isiga, isigf, tau, f, agent, turn="agent", aab=[], observ=[]):
        #if a "reset" is done, then we always use "client" turn
        reset=False
        if turn=="creset":
            turn="client"
            reset=True
            isigf = isigf["agent"] # Note the type of isigf is changed here. TODO Fix it
        else:
            isigf = isigf[turn]

        #now we will insert aab into f here
        #this seems like a hack, but in fact it is only a small one
        #we are "using" the "b" slot in f to hold what the agent action is
        #and when turn=="agent", the variance on b is tiny, so forces
        #the samples to have a Fb-value which is the same as aab
        #here, we could also replace f[0:3] with aab if aab is 6-D :
        #f=NP.concatenate((aab,f[6:9]))
        #this would happen if the agent can also set its own identity with an action
        #the isigf would also need to be modified to reflect this
        #see in the constructor the comment about this - search for AAB6D
        if turn=="agent" and not aab==[]:
            f=NP.concatenate((f[0:3],aab,f[6:9]))
    
        #special case - observation is used as f_b directly if there was a reset
        if reset and not observ==[]:
            f=NP.concatenate((f[0:3],observ,f[6:9]))

        #get the instantiated versions of H and C - these have the actual values from tau
        if turn=="agent":
           (self.h, self.c) = agent.agentMappings.getHC(tau);
        elif turn=="client": 
           (self.h, self.c) = agent.clientMappings.getHC(tau);

    
        self.c = self.c.transpose()

        self.weight = self._computeWeight()

        (self.mu_n, self.sig_n) = self._computeMuSig(f, isiga, isigf)

    def sampleNewFVar(self):
        self.f_s = NP.random.multivariate_normal(self.mu_n, self.sig_n)
        self.f_s = self.f_s.transpose()
    
        return self.f_s



#gets the optimal F value if only using the affect control principle, not the inertia term
def getDefaultMeanF(fvars,tvars,H,C,tau,f,turn="agent",aab=[],observ=[]):


    #now we will insert aab into f here
    #this seems like a hack, but in fact it is only a small one
    #we are "using" the "b" slot in f to hold what the agent action is
    #and when turn=="agent", the variance on b is tiny, so forces
    #the samples to have a Fb-value which is the same as aab
    #here, we could also replace f[0:3] with aab if aab is 6-D :
    #f=NP.concatenate((aab,f[6:9]))
    #this would happen if the agent can also set its own identity with an action
    #the isigf would also need to be modified to reflect this
    #see in the constructor the comment about this - search for AAB6D
    if turn=="agent" and not aab==[]:
        f=NP.concatenate((f[0:3],aab,f[6:9]))


    #get the instantiated versions of H and C - these have the actual values from tau
    (tmpH,tmpC)=instHCfromTau(tvars,H,C,tau,turn)


    theweight=0.0

    zmat= NP.zeros((3,9))
    K= NP.eye(9)-NP.vstack([zmat, tmpH, zmat]).transpose()

    mu_n=NP.dot(NP.linalg.inv(K),tmpC)

    return mu_n







#multi-variate normalpdf function (unused)
def mvar_normpdf(x, mean, cov):
    size=len(x)
    icov=NP.linalg.inv(cov)
    dm=NP.asarray(x-mean)
    md=NP.transpose(dm)
    num=-0.5*NP.dot(NP.dot(md,icov),dm)
    thedet=NP.linalg.det(cov)
    denom=( math.pow((2*NP.pi),float(size)/2) * math.pow(thedet,1.0/2) )
    return num-math.log(denom)


#computes the mean of an unweighted array of States
def sampleMeanUnweighted(samples):
    i=0
    avgstate = samples[0].get_copy()
    for s in samples[1:]:
        avgstate +=  s
    avgstate = avgstate/len(samples)
    return avgstate

#computes the mean of a weighted array of states
def sampleMean(samples):

    sumw = reduce(lambda x,y: x+float(y.weight), samples, 0.0)
    avgstate = samples[0].apply_weight()
    if sumw:
        f = lambda x, y: x + y.apply_weight()
        avgstate = reduce(f, samples, State.zero())

        avgstate = avgstate/sumw
    else:
        avgstate=sampleMeanUnweighted(samples)
    return avgstate


#an obsSet is a combination of two elements -
#First, a 3D affective vector and
#Second, the the propositional observation,
#which is actually class dependent
#but needs to have operators for addition and division-by-a-scalar defined
#the division operator / will divide each element of the propositional observation
def getMeanObs(obsSet):
    if obsSet:
        avgobs=NP.array(obsSet[0][0])
        avgpobs=obsSet[0][1]
        for o in obsSet[1:]:
            avgobs=avgobs+o[0]
            avgpobs=avgpobs+o[1]
        avgobs=avgobs/len(obsSet)
        avgpobs=avgpobs/len(obsSet)
        return (avgobs,avgpobs)
    else:
        return []

#compute the mean of a set of fsamples
def fsampleMean(fsamples):
    if fsamples:
        avgf = fsamples[0]
        for f in fsamples[1:]:
            avgf = avgf+f
        avgf = avgf/len(fsamples)
        return avgf
    else:
        return []

#raw squared distance between two epa vectors
def raw_dist(epa1,epa2):
    dd = NP.array(epa1)-NP.array(epa2)
    return NP.dot(dd,dd)


#maps a sentiment label cact to an epa value using the dictionary fbeh
def mapSentimentLabeltoEPA(fbeh,cact):
    return map(lambda x: float (x), [fbeh[cact]["e"],fbeh[cact]["p"],fbeh[cact]["a"]])

#raw distance from an epa vector oval to a behaviour vector
def fdist(oval,fbehv):
    cd=[float(fbehv["e"]),float(fbehv["p"]),float(fbehv["a"])]
    return raw_dist(oval,cd)

#identify if at least one match exists between two lists of booleans
def one_match(lob1,lob2):
    return reduce(lambda x,y: y[0] and y[1] or x, zip(lob1,lob2), False)
    

#find label in dictionary fdict with epa vector closest to epa-vector oval
def findNearestEPAVector(oval,fdict,inst_list=None):
    start=True
    mind=0
    for f in fdict:
        if inst_list==None or one_match(fdict[f]["institutions"][2:11],inst_list):
            fd = fdist(oval,fdict[f])
            if start or fd<mind:
                mind=fd
                bestf=f
                start=False
    return bestf

#find nearest behaviour to epa vector oval in the dictionary fbeh
#if inst_list is not None, find one 
#that is compatible with the institutions in inst_list (a list of booleans)
#fdict[beh]["institutions"] must carry the list of booleans for each beh
#and at least one value of inst_list must match one value in fdict[beh]["institutions"]
def findNearestBehaviour(oval,fbeh, inst_list=None):
    return findNearestEPAVector(oval,fbeh,inst_list)


#inserts ld=(label,dist) into list of such pairs
#up to a maximum of N, sorted by increasing dist
def insert_into(lds,ld,N):
    i=0
    while i<N and ld[1]>lds[i][1]:
        i += 1
    while i<N:
       keepit=lds[i]
       lds[i]=ld
       ld=keepit
       i += 1
    return lds

#find N nearest behaviours in database to oval
def findNearestBehaviours(oval,fbeh,N):
    mind=0
    bestbs=zip(N*[''],N*[1000.0])
    for f in fbeh:
        fd = fdist(oval,fbeh[f])
        bestbs=insert_into(bestbs,(f,fd),N)
    return map(lambda x: x[0],bestbs)

#find nearest identity in dictionary fid to epa-value oval
def findNearestIdentity(oval,fid):
    return findNearestEPAVector(oval,fid)


#draw an unweighted set of samples from the weighted set in input samples
#if N is -1, resample the same number, otherwise sample N new samples
def drawSamples(samples,N=-1):
    if N==-1:
        N=len(samples)

    weights=map(lambda x: x.weight,samples)
    cumweights=NP.cumsum(weights);
    cumw=cumweights/cumweights[-1]
    #don't actually need to do this - can just select
    #one random point in the first interval 1..(1/N)
    #and then uniform multiples of 1/N from there
    rsn=NP.random.random_sample((N,))
    rsns=NP.sort(rsn)

    i=0
    new_samples=[]
    for rs in rsns:
        while cumw[i]<rs:
            i+=1
        thesample=samples[i].get_copy()
        thesample.weight=1.0
        new_samples.append(thesample)
    return new_samples



#print all samples
def printSamples(samples):
    map(lambda x: x.print_val(),samples)

#print all samples, no decorations
def printRawSamples(samples,fp=None):
    map(lambda x: x.print_raw_val(fp),samples)

#print only f and w values of all samples
def printSamplesFW(samples):
    map(lambda x: x.print_fw(),samples)

#only print samples with a weight greater than heavy
def printHeavySamples(samples,heavy):
    map(lambda x: x.print_val(), filter(lambda x: x.weight>heavy, samples))


def getFactor(indices, tau):
    return reduce(lambda x, i: x*tau[i], indices, 1)

def swapActorClientValues(vector):
    return NP.concatenate((vector[6:9], vector[3:6], vector[0:3]))

def swapActorClientTdyn(tdyn):
    return map(lambda x: [swapACVars(x[0]), swapActorClientValues(x[1])], tdyn)

class TauMappings:
    def __init__ (self, tdyn, tvars):
        self.gTemplateTbe = [];
        self.gTemplateTbp = [];
        self.gTemplateTba = [];
        self.gTemplateTau = [];

        self.mTbe = [];
        self.mTbp = [];
        self.mTba = [];
        self.mTau = [];

        self.tau = NP.zeros(9);

        self.H = NP.ndarray(shape=(3,9));
        self.C = NP.ndarray(shape=(1,9));

        for col in tdyn:
            if has_variable(col[0], "Tbe"):
               self.gTemplateTbe += [map(lambda x: tvars.index(x), filter(lambda y: y != "Tbe" , col[0]))]
               self.mTbe += [col[1]];
            elif has_variable(col[0], "Tbp"):
               self.gTemplateTbp += [map(lambda x: tvars.index(x), filter(lambda y: y != "Tbp" , col[0]))]
               self.mTbp += [col[1]];
            elif has_variable(col[0], "Tba"):
               self.gTemplateTba += [map(lambda x: tvars.index(x), filter(lambda y: y != "Tba" , col[0]))]
               self.mTba += [col[1]];
            else:
               self.gTemplateTau += [map(lambda x: tvars.index(x) , col[0])]
               self.mTau += [col[1]];

#    def getM(self):
        

#    def getTau(self):

    def computeG(self, tau):
            self.gTbe = map(lambda i: getFactor(i, tau), self.gTemplateTbe)
            self.gTbp = map(lambda i: getFactor(i, tau), self.gTemplateTbp)
            self.gTba = map(lambda i: getFactor(i, tau), self.gTemplateTba)
            self.gTau = map(lambda i: getFactor(i, tau), self.gTemplateTau)

    def getHC(self, tau):
      #  if ((self.tau != tau).any()):
            self.tau = tau;

            self.gTbe = map(lambda i: getFactor(i, tau), self.gTemplateTbe)
            self.gTbp = map(lambda i: getFactor(i, tau), self.gTemplateTbp)
            self.gTba = map(lambda i: getFactor(i, tau), self.gTemplateTba)
            self.gTau = map(lambda i: getFactor(i, tau), self.gTemplateTau)

            self.H[0] = NP.dot(self.gTbe, self.mTbe)
            self.H[1] = NP.dot(self.gTbp, self.mTbp)
            self.H[2] = NP.dot(self.gTba, self.mTba)
            self.C[0] = NP.dot(self.gTau, self.mTau)

            return (self.H, self.C)
        

#---------------------------------------------------------------------------------------------------------------------------
#class to describe the state - including fundamental and transient sentiments and x
#this will be used for sampling, so includes a weight
#note that the "State" will change in subclasses of "Agent", but this is defined on-the-fly by simply
#initialising the "x" component to a certain size array.  However, the "x" component must always contain
#the "turn" as the first element.
#---------------------------------------------------------------------------------------------------------------------------

class State(object):

    turnnames=["agent","client"]

    @staticmethod
    def zero():
        return State(NP.zeros(9), NP.zeros(9), [0], 0)

    def __init__(self,f,tau,x,weight):
        #fundamentals
        self.f=f
        #transients
        self.tau=tau
        #propositional state
        #x is a vector, and *must* contain the turn as the first element
        self.x = x
        #sample weight
        self.weight=weight

    def invert_turn(self):
        return self.turnnames.index(invert_turn(self.get_turn()))

    #operators that must be defined
    def __add__(self,other):
        return State(self.f+other.f,self.tau+other.tau,self.x+other.x,self.weight+other.weight)
    def __mul__(self,other):
        return State(self.f*other.f,self.tau*other.tau,self.x*other.x,self.weight+other.weight)
    def __div__(self,other):
        return State(NP.divide(self.f,other),NP.divide(self.tau,other),NP.divide(self.x,other),NP.divide(self.weight,other))

    def __iadd__(self,other):
        self.f=NP.add(self.f,other.f)
        self.tau=NP.add(self.tau,other.tau)
        self.x=NP.add(self.x,other.x)
        self.weight=NP.add(self.weight,other.weight)
        return self

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
           return False
        retval = True
        retval = retval & (self.f == other.f).all()
        retval = retval & (self.tau == other.tau).all()
        retval = retval & (self.weight == other.weight)
        retval = retval & (self.x == other.x)
        return retval

    def __ne__(self, other):
        return not self.__eq__(other)

    #make a copy and return it
    def get_copy(self):
        return State(self.f,self.tau,self.x,self.weight)

    #apply the weight and return a new state with each element scaled by the weight
    def apply_weight(self):
        return State(NP.dot(self.f,self.weight),NP.dot(self.tau,self.weight),NP.dot(self.x,self.weight),self.weight)

    #State *must* implement get_turn
    #this must produce either "agent" or "client"
    def get_turn(self):
        return self.turnnames[int(round(self.x[0]))]

    #add some noise to samples ("roughening" - see Gordon et al. 1993 or Chapt. 10 of Nando's book)
    def roughen(self,noiseVector):
        nois=(2*NP.random.random((1,9))-1.0)*noiseVector
        self.f = self.f + nois.flatten(1)
        #nois=(2*NP.random.random((1,9))-1.0)*noiseVector
        #self.tau = self.tau + nois.flatten(1)

    def get_fw_as_string(self):
        theval = "f :"+str(self.f)+"\n"
        theval += "weight :"+str(self.weight)+"\n"
        return theval

    def get_raw_val_as_string(self):
        theval = ""
        theval += str(self.weight)+" "
        for x in self.f:
            theval += str(x)+" "
        for x in self.tau:
            theval += str(x)+" "
        for x in self.x:
            theval += str(x)+" "
        return theval
        

    def get_val_as_string(self):
        theval = 50*"~"+"\n"
        theval += self.get_fw_as_string()
        theval += "tau :"+str(self.tau)+"\n"
        theval += "x: "+str(self.x)+"\n"
        theval += 50*"~"+"\n"
        return theval

    def print_val(self,fp=None):
        theval = self.get_val_as_string()
        if fp == None:
            print theval
        else:
            fp.write(theval)
            
    def print_raw_val(self,fp=None):
        theval = self.get_raw_val_as_string()
        if fp == None:
            print theval
        else:
            fp.write(theval)
            fp.write("\n")
            
    def print_fw(self,fp=None):
        theval = self.get_fw_as_string()
        if fp == None:
            print theval
        else:
            fp.write(theval)


#A simple class to help with initialisation fo samples
class SampleInitialiser(object):
    #---------------------------------------------------------------------------
    #constructor
    #---------------------------------------------------------------------------
    def __init__(self,*args,**kwargs):
        #parameter of the deflection
        #sort of stateless really but I could move some stuff from the functions below to here
        #I guess you need something here
        False
        
    #initialise a set of samples \fsb (9D with fub_a,fub_b,fub_c 3D vectors)
    #agent_ids and client_ids are arrays of 4-tuples, where the first element gives
    #the proportion of that id, and the second element gives the id as a label or 3D vector
    #and the third element gives the variance for that id, and the fourth, if present gives the gender (otherwise assumed to be agent_gender)
    #the samples in fsb are drawn agent_ids and client_ids
    #if agent_ids or client_ids is [] then the mean_id and cov_id are used for a single Gaussian
    #initx is the distribution of the initial turn [agent,client] - e.g. [1,0] means its definitely agent's turn
    #while [0.5,0.5] means the turn is unknown
    def initialise_samples(self,n,fifname,agent_gender,agent_ids=[],client_ids=[],initx=[0.5,0.5],mean_id=[],cov_id=[]):
        

        a_ids = copy.deepcopy(agent_ids) 
        c_ids = copy.deepcopy(client_ids)
        
        if a_ids==[] or c_ids==[]: 
            if mean_id==[] or cov_id==[]:
                (mean_id,cov_id)=getIdentityStats(fifname,agent_gender)
            if a_ids==[]:
                a_ids=[[1.0,mean_id,cov_id,agent_gender]]
            if c_ids==[]:
                c_ids=[[1.0,mean_id,cov_id,agent_gender]]


        #now, a_ids becomes another 4-tuple but with all id labels replaced with 
        #their EPA values. Gender is not added if it is not present originally in a_ids or c_ids
        a_ids=make_ids_covs(fifname,a_ids,agent_gender)
        c_ids=make_ids_covs(fifname,c_ids,agent_gender)
        return self.initialise_samples_help(n,a_ids,c_ids,initx)

    def initialise_samples_uniform(self,n,initx=[0.5,0.5],mv=[0,0,0],minv=-4.3,maxv=4.3):
        samples=[]
        diffv=maxv-minv
        for i in range(n):
            fuba = NP.asarray(mv)+NP.random.random(3)*diffv+minv
            fubc = NP.asarray(mv)+NP.random.random(3)*diffv+minv
            fubb=[0.0,0.0,0.0]  #is never used so is not important
            tmpfv = NP.hstack([fuba,fubb,fubc])
            tmpix = self.initialise_sample_x(initx)
            samples.append(State(tmpfv,tmpfv,tmpix,1.0))
        return samples
            


    #initialises a set of n samples from agent_ids and client_ids, where 
    #agent_ids and client_ids are arrays of 3-tuples, where the first element gives
    #the proportion of that id, and the second element gives the id as a 3D vector
    #and the third element gives the covariance for that id
    #agent_ids and client_ids must each have at least one element
    #initx is the initial turn ("agent" or "client")
    def initialise_samples_help(self,n,agent_ids,client_ids,initx):
        samples=[]
            
        assert len(agent_ids)>0
        assert len(client_ids)>0

        agent_propor=map(lambda x: x[0],agent_ids)
        client_propor=map(lambda x: x[0],client_ids)
    

        #draw the samples
        for i in range(n):
            #choose identities for each
            ai=list(NP.random.multinomial(1, agent_propor)).index(1)
            ci=list(NP.random.multinomial(1, client_propor)).index(1)
            #draw the sample
            fuba=NP.random.multivariate_normal(agent_ids[ai][1],agent_ids[ai][2])
            fubc=NP.random.multivariate_normal(client_ids[ci][1],client_ids[ci][2])
            fubb=[0,0,0]  #is never used so is not important
            tmpfv = NP.hstack([fuba,fubb,fubc])
            tmpix = self.initialise_sample_x(initx)
            samples.append(State(tmpfv,tmpfv,tmpix,1.0))
        return samples

    def initialise_sample_x(self, initx):
        return [list(NP.random.multinomial(1,initx)).index(1)]

#---------------------------------------------------------------------------------------------------------------------------
#The main Agent Class
#this can be used directly or subclassed (see DiscreteTutoringAgent in discretetutor.py )
#subclasses of Agent basically implement different values of "X", which is always an array of size at least 1
#with the first element being a binary "turn" with x[0]=0 meaning "agent" and x[0]=1 meaning client
#this relationship (0=agent and 1=client) should be defined by the class variable "turnnames" in class State
#---------------------------------------------------------------------------------------------------------------------------
class Agent(object):
    #---------------------------------------------------------------------------
    #constructor
    #---------------------------------------------------------------------------
    def __init__(self,*args,**kwargs):
        #parameter of the deflection
        self.alpha_value=kwargs.get("alpha_value",1.0)

        #parameter of the fundamentals identity prior
        #this first one governs how much variance we expect in the identities - should normally be small
        #if  we don't expect identities to change much
        #beta_value_agent should be set to 1e-20 if we are using a 6-D aab vector
        #the agent is in charge of setting its own identity at each iteration
        self.beta_value_agent=kwargs.get("beta_value_agent",0.01)
        self.beta_value_client=kwargs.get("beta_value_client",0.01)
        self.beta_value_agent_init=kwargs.get("beta_value_agent_init",0.01)
        self.beta_value_client_init=kwargs.get("beta_value_client_init",0.01)

        #for making identity-setting actions search for AAB6D

        #this governs the variance in behaviours - should normally be large in fact infinite, but this causes all sorts of
        #problems so we make it really really big
        self.beta_valueb_client=kwargs.get("beta_valueb_client",1e+12)
        self.beta_valueb_client_init=kwargs.get("beta_valueb_client_init",1.0)
        self.beta_valueb_agent=kwargs.get("beta_valueb_agent",1e-12)
        self.fifname=kwargs.get("identities_file","fidentities.dat")
        self.fbfname=kwargs.get("behaviours_file","fbehaviours.dat")

        #parameter of the observation function - should be related to o_hitrate below
        self.gamma_value=kwargs.get("gamma_value",1.0)

        #discount factor - used for POMCP planning
        self.discount_factor=kwargs.get("discount_factor",0.9)

        #this is not used anymore
        self.beliefSimilarity = -1.0

        #number of samples to use
        self.N=kwargs.get("N",300)

        #this is never used - it doesn't matter what the gender of the client is
        #since the dynamics and lexicons are based solely on the agent gender
        self.client_gender=kwargs.get("client_gender","male")
        #agent gender matters
        self.agent_gender=kwargs.get("agent_gender","male")

        self.agent_rough=kwargs.get("agent_rough",0.0)
        self.client_rough=kwargs.get("client_rough",0.0)

        self.init_turn=kwargs.get("init_turn","agent")

        #number of samples of f to use when computing average f prediction for default action
        self.nsamples_avgf=kwargs.get("N_avgf",100)


        self.use_pomcp=kwargs.get("use_pomcp",False)
        self.pomcp_doInteractive = kwargs.get("pomcp_interactive",False)

        #parameters specific to pomcp
        self.alpha_value_pomcp=kwargs.get("alpha_pomcp",0.1)
        self.gamma_value_pomcp = kwargs.get("gamma_pomcp",0.001)

        self.fixed_policy=kwargs.get("fixed_policy",None)

        #some defaults -
        self.x_avg=0
        self.deflection_avg=0

        #samples
        self.samples=[]

        #POMCP parameters
        self.pomcp_agent=None
        #number of continuous and discrete actions
        self.numDiscActions=kwargs.get("numdact",1)
        self.numContActions=kwargs.get("numcact",1)
        self.obsResolution = kwargs.get("obsres",0.1)
        self.actResolution = kwargs.get("actres",0.1)
        self.pomcpTimeout = kwargs.get("pomcp_timeout",1.0)

        #initialisation routines
        self.init_covariances()
        self.init_sentiment_dictionaries(self.fifname,self.fbfname)

        self.agentMappings  = TauMappings(self.tdyn, tvars)
        self.clientMappings = TauMappings(swapActorClientTdyn(self.tdyn), tvars)


        self.verbose = True

    def print_params(self):
        if self.verbose:
            print "alpha_value: ",self.alpha_value
            print "beta_value_agent: ",self.beta_value_agent
            print "beta_value_client: ",self.beta_value_client
            print "beta_value_client_init: ",self.beta_value_client_init
            print "beta_value_agent_init: ",self.beta_value_agent_init
            print "beta_valueb_client: ",self.beta_valueb_client
            print "beta_valueb_agent: ",self.beta_valueb_agent
            print "gamma_value: ",self.gamma_value
            print "discount_factor: ",self.discount_factor
            print "N: ",self.N
            print "client_gender: ",self.client_gender
            print "agent_gender: ",self.agent_gender
            print "agent_rough: ",self.agent_rough
            print "client_rough: ",self.client_rough
            print "use_pomcp?: ",self.use_pomcp
            if self.use_pomcp and not self.pomcp_agent==None:
                self.pomcp_agent.printParams()
            print "alpha_value_pomcp: ",self.alpha_value_pomcp
            print "gamma_value_pomcp: ",self.gamma_value_pomcp


    #---------------------------------------------------------------------------
    #initialisation functions
    #---------------------------------------------------------------------------
    def init_covariances(self):
        #set up the covariance matrices
        e3=NP.eye(3)
        z3=NP.zeros((3,3))
        betabc2=self.beta_valueb_client**2
        betabci=self.beta_valueb_client_init**2
        betaba2=self.beta_valueb_agent**2
        betaa2=self.beta_value_agent**2
        betac2=self.beta_value_client**2
        betaci=self.beta_value_client_init**2
        betaai=self.beta_value_agent_init**2
        sigfa1 = NP.vstack([NP.hstack([e3, z3, z3]),NP.hstack([z3, z3, z3]),NP.hstack([z3, z3, z3])])*betaa2
        sigfai = NP.vstack([NP.hstack([e3, z3, z3]),NP.hstack([z3, z3, z3]),NP.hstack([z3, z3, z3])])*betaai
        sigfc1 = NP.vstack([NP.hstack([z3, z3, z3]),NP.hstack([z3, z3, z3]),NP.hstack([z3, z3, e3])])*betac2
        sigfci = NP.vstack([NP.hstack([z3, z3, z3]),NP.hstack([z3, z3, z3]),NP.hstack([z3, z3, e3])])*betaci
        sigf2bc = NP.vstack([NP.hstack([z3, z3, z3]),NP.hstack([z3, e3, z3]),NP.hstack([z3, z3, z3])])*betabc2
        sigf2bci = NP.vstack([NP.hstack([z3, z3, z3]),NP.hstack([z3, e3, z3]),NP.hstack([z3, z3, z3])])*betabci
        sigf2ba = NP.vstack([NP.hstack([z3, z3, z3]),NP.hstack([z3, e3, z3]),NP.hstack([z3, z3, z3])])*betaba2

        sigfb = e3*betabc2
        isigfb=e3/betabc2

        self.sigf={}
        self.isigf={}
        self.sigf_unconstrained_b={}
        self.isigf_unconstrained_b={}



        #sigfa1: betaa2 : beta_value_agent^2 over agent id  (e.g. 0.001)
        #sigfc1: betac2 : beta_value_client^2 over client id (e.g. 0.001)
        #sigf2bc: betabc2 : beta_valueb_client^2 over behaviour  (e.g. 1e+20 or unconstrained)
        self.sigf["client"]=sigfa1+sigfc1+sigf2bc
        self.isigf["client"] = NP.linalg.inv(self.sigf["client"])

        #sigf2ba: betaba2 : beta_valueb_agent^2 over behaviour (e.g. 1e-20 or constrained to be agent action)
        self.sigf["agent"]=sigfa1+sigfc1+sigf2ba
        self.isigf["agent"] = NP.linalg.inv(self.sigf["agent"])


        #for agent with unconstrained behaviour
        self.sigf_unconstrained_b["agent"]=sigfa1+sigfc1+sigf2bc
        self.isigf_unconstrained_b["agent"] = NP.linalg.inv(self.sigf_unconstrained_b["agent"])

        #for client with unconstrained beahaviour (is the same as above right now, but could be different)
        self.sigf_unconstrained_b["client"]=sigfa1+sigfc1+sigf2bc
        self.isigf_unconstrained_b["client"] = NP.linalg.inv(self.sigf_unconstrained_b["client"])


        self.sigfi=sigfai+sigfci+sigf2bci
        self.isigfi = NP.linalg.inv(self.sigfi)


        alpha2=self.alpha_value**2
        self.siga=NP.eye(9,9)*alpha2
        self.isiga=NP.eye(9,9)/alpha2
        self.sigab = e3*alpha2
        self.isigab = e3/alpha2

        self.isiga_pomcp=NP.eye(9,9)/(self.alpha_value_pomcp**2)

        self.init_output_covariances()

    #needs to be a separate function, because we may want to override this in applications/subclasses
    def init_output_covariances(self):

        self.gamma_value2=self.gamma_value*self.gamma_value

        #precomputed stuff for computing normal pdfs for 3D vectors
        theivar=1.0/float(self.gamma_value2)
        thedet=math.pow(float(self.gamma_value),3)
        self.ldenom=math.log(math.pow((2*NP.pi),1.5)*thedet)
        self.theivar = theivar*0.5



    def init_sentiment_dictionaries(self,fifname,fbfname=[]):

        self.fidentities_client=readSentiments(fifname,self.agent_gender)
        self.fidentities_agent=readSentiments(fifname,self.agent_gender)
        
        if not fbfname==[]:
            self.fbehaviours_agent = readSentiments(fbfname,self.agent_gender)

        if self.agent_gender=="female":
            self.tdyn=tdyn_female
        else:
            self.tdyn=tdyn_male


    def init_HC(self):
        #now we construct the functions that will compute the H and C matrices
        #which is 9x3 with the jth column giving the terms of M't that contain
        #B_{d_a(j)} and C is a 9x1 vector that gives the terms of M't that contain
        #no B variable.  M and C are functions of tau and x' (the turn)
        #the turn (x') is necessary as we have to swap all A and C variables
        #if the current turn (x) is the client (so x'=agent)
        Hagent={}
        for j in ["Tbe","Tbp","Tba"]:
            Hagent[j]=filter(lambda xx: has_variable(xx[0],j),self.tdyn)

        Cagent=filter(lambda xx: not has_variable(xx[0],"Tb"), self.tdyn)

        Hclient={}
        for j in ["Tbe","Tbp","Tba"]:
            Hclient[j]=swapAC(filter(lambda xx: has_variable(xx[0],j),self.tdyn))
        Cclient=swapAC(filter(lambda xx: not has_variable(xx[0],"Tb"), self.tdyn))


        self.H={}
        self.H["client"]=Hclient
        self.H["agent"]=Hagent

        self.C={}
        self.C["client"]=Cclient
        self.C["agent"]=Cagent

        #create version that already figure out the mapping from tvars to indices
        #and have removed all values of b
        self.iC={}
        self.iC["agent"]=zip(map(lambda x: map(lambda y: tvars.index(y),x[0]),self.C["agent"]),map(lambda x: x[1],self.C["agent"]))
        self.iC["client"]=zip(map(lambda x: map(lambda y: tvars.index(y),x[0]),self.C["client"]),map(lambda x: x[1],self.C["client"]))

        self.iH={}
        self.iH["agent"]={}
        self.iH["client"]={}
        for j in ["Tbe","Tbp","Tba"]:
            self.iH["agent"][j]=zip(map(lambda x: map(lambda y: tvars.index(y),filter(lambda z: not z[1]=="b",x[0])),self.H["agent"][j]),map(lambda x: x[1],self.H["agent"][j]))
            self.iH["client"][j]=zip(map(lambda x: map(lambda y: tvars.index(y),filter(lambda z: not z[1]=="b",x[0])),self.H["client"][j]),map(lambda x: x[1],self.H["client"][j]))

    #this is where "x" is actually defined - here we set it to simply the "turn"
    def initialise_x(self,initx):
        return [State.turnnames.index(initx)]




    def initialise(self,f,initx):
        #draw a set of samples from the initial distributions
        self.samples=[]
        for i in range(self.N):
            tmpfv=NP.random.multivariate_normal(f,self.sigfi)
            tmpix=self.initialise_x(initx)
            xx=State(tmpfv,tmpfv,tmpix,1.0)
            self.samples.append(xx)
        avgs=sampleMean(self.samples)
        self.initialise_pomcp(self.samples)
        self.x_avg=avgs.x
        return avgs


    def set_samples(self,samples):
        self.samples=[]
        for s in samples:
            self.samples.append(s.get_copy())
        avgs=sampleMean(self.samples)
        return avgs

    def initialise_array(self,f,prop,initx):
        #draw a set of samples from an array of initial distributions with weights given by prop
        #the set is drawn according to prop so f can be of any size greater than prop
        self.samples=[]
        nums=NP.random.multinomial(self.N, prop)
        j=0
        self.init_f=f
        self.init_prop=prop
        self.init_initx=initx
        for n in nums:
            for i in range(n):
                tmpfv=NP.random.multivariate_normal(f[j],self.sigfi)
                tmptv=NP.random.multivariate_normal(f[j],self.sigfi)
                tmpix=self.initialise_x(initx)
                xx=State(tmpfv,tmpfv,tmpix,1.0)    #should this be State(tmpfv,tmptv,tmpix,1.0) ????
                self.samples.append(xx)
            j += 1
        avgs=sampleMean(self.samples)
        self.x_avg=avgs.x
        self.initialise_pomcp(self.samples)
        return avgs

    #POMCP intitialisation
    def initialise_pomcp(self,beliefState):
        if self.use_pomcp:
            self.pomcp_agent=POMCP(numcact=self.numContActions,numdact=self.numDiscActions,obsres=self.obsResolution,actres=self.actResolution,timeout=self.pomcpTimeout,verbose=self.verbose)
            self.pomcp_agent.POMCP_eval(beliefState,100,self)



    #---------------------------------------------------------------------------
    #sampling functions
    #---------------------------------------------------------------------------
    def sample_next_f(self, state, N=100):
        #draw a set of samples over the next fundamental sentiment
        fvsamples=[]
        fsampler = FVarSampler(self.isiga, self.isigf_unconstrained_b, \
                             state.tau, state.f, self, state.get_turn())
        for x in range(N):
            fv = fsampler.sampleNewFVar()
            fv = fv.transpose()
            fvsamples.append(fv)

        avgf = fsampleMean(fvsamples)
        return avgf


    #default value simply inverts the turn
    #overload in subclass
    def sampleXvar(self,f,tau,state,aab,paab=None):
        return [state.invert_turn()]

    #overload in subclass
    def sampleXObservation(self,s):
        return s.x


    #draw a next sample from a currrent one "state" by sampling f, tau and x in that order
    #turn can eventually be removed from the list of arguments, but left in now so other things keep working
    def sampleNext(self, fVarSampler, fvars, tvars, state, aab, paab=None):
        h   = fVarSampler.h
        c   = fVarSampler.c
        wgt = fVarSampler.weight

        fsample = fVarSampler.sampleNewFVar()

        #sample from T using the H and C matrices we computed from
        #    tau in sampleNewFVar
        #TODO: Move these functions into (renamed) FVarSample or
        #      new similar class
        tsample = sampleTvars(tvars, h, c, fsample)

        #this is class dependent
        xsample = self.sampleXvar(fsample,tsample,state,aab,paab)

        return State(fsample, tsample, xsample, wgt)


    #function that must be overloaded in any subclass
    #here it is just a check on the turn
    def evalSampleXvar(self,sample,xobs,oldsample=[]):
        #check that turn is the same only
        #here we compare to the newsample ,but could do oldsample instead
        if sample.x[0] == xobs[0]:
            return 1.0
        else:
            return 0.0

    #this will be used in the emotional agent class
    def evalSampleEvar(self,state,turn,eobserv):
        return 0.0
        

    #add some noise to samples ("roughening" - see Gordon et al. 1993 or Chapt. 10 of Nando's book)
    def roughenSamples(self,samples):
        noiseVector=NP.concatenate((NP.ones((3,1))*self.agent_rough,NP.zeros((3,1)),NP.ones((3,1))*self.client_rough)).flatten(1)
        map(lambda x: x.roughen(noiseVector),samples)

    #compute the log-PDF of a normal with mean mean and
    #constant multiplicative factor in the exponent ivar,
    #and log-denominator ldenom
    #mean is a scalar value (usually zero)
    def normpdf(self,x, mean, ivar, ldenom):
        num=0.0
        for xv in x:
            num = num-ivar*( float(xv) - float(mean) )**2
            
        return num-ldenom



    #evaluates a sample of Fvar'=state.f
    def evalSampleFvar(self,fvars,tdyn,state,ivar,ldenom,turn,observ):
        weight=0.0
        if (not observ==[]) and turn=="client":
            fb=getbvars(fvars,state.f)
            dvo=NP.array(fb)-NP.array(observ)
            weight += self.normpdf(dvo,0.0,ivar,ldenom)
        else:
            weight=0.0
        return weight

    #the main SMC update step. Given an affective action aab and a propositional action paab,
    #and an affective observation observ and an  observation xobserv, update the samples
    #by first drawing an unweighted set, propagating those forward, computing the weights
    #and finally computing the deflection
    #eobserv is used in EmotionalAgent (see bayesactemot.py) as observation of emotional state
    def propagate_forward(self,aab,observ,xobserv=[],paab=None,verb=False,plotter=None,agent=eTurn.learner,eobserv=None):

        if False and verb:
            print 100*"!","samples before resample:"
            printSamples(self.samples)

        #resample to get an unweighted set
        new_samples=drawSamples(self.samples)

        #add "jitter" here instead of the proposal idea...
        #possibly just add the jitter to the client identity
        if self.agent_rough>0 or self.client_rough>0:
            self.roughenSamples(new_samples)

        #this is where the samples can be extracted for plotting
        if self.verbose and verb:
            print 100*"!","unweighted set:"
            printSamples(new_samples)

        #send new samples to plotter
        if (None != plotter):
            self.sendSamplesToPlotter(new_samples,plotter,agent)
            
            if (eTurn.learner == agent):
                plotter.m_LearnerPreviousAction = aab
            else:
                plotter.m_SimulatorPreviousAction = aab

        #propagate forward
        self.samples=[]
        totweight=0.0
        for sample in new_samples:
            fVarSampler = FVarSampler(self.isiga, self.isigf, sample.tau, \
                                      sample.f, self, sample.get_turn(), \
                                      aab, observ)

            newsample = self.sampleNext(fVarSampler, fvars, tvars, sample, \
                                        aab, paab)

            theweight = self.evalSampleFvar(fvars,self.tdyn,newsample,self.theivar,self.ldenom,sample.get_turn(),observ)
            newweight = newsample.weight+theweight

            #emotional agent stuff, not used in base Agent class
            theweight = self.evalSampleEvar(newsample,sample.get_turn(),eobserv)
            newweight = newweight+theweight
            thexweight = self.evalSampleXvar(newsample,xobserv,sample)


            newsample.weight=math.exp(newweight)*thexweight
            totweight += newsample.weight
            self.samples.append(newsample)

        if self.verbose:
            print 20*"%"," total weight: ",totweight
        number_of_reset_attempts=0
        while totweight<1e-20:
            #reset - the total weight has collapsed and the particle filter is broken
            if self.verbose:
                print 100*"r","eset"
            #we used to do it this way
            #tmp_avgs=self.initialise_array(self.init_f,self.init_prop,self.init_initx)
            #now, draw a brand new set of samples, but using observ as if it was set exactly
            #can do this using turn="reset" (special case)
            self.samples=[]
            totweight=0.0
            for sample in new_samples:
                fVarSampler = FVarSampler(self.isiga, self.isigf, sample.tau, \
                                          sample.f, self, "creset", \
                                          aab, observ)

                newsample = self.sampleNext(fVarSampler, fvars, tvars, sample, \
                                            aab, paab)
                #here we don't compute the weight at all using evalSampleFvar since it may be totally wrong
                theweight = 0.0
                newweight = newsample.weight+theweight

                thexweight = self.evalSampleXvar(newsample,xobserv,sample)

                #now, sometimes this fails and returns zero if all the "turn" indicators in the previous
                #sample disagree with xobserv, and we are using an evaluation of the turn where the previous
                #sample is used as the X value for the weight.  In such cases, we will assume that at least
                #some of the new sample x values are "correct" and we'll set the weight to acccordingly
                #or we can just set all weights to one and hope for the best
                if thexweight < 1e-20:
                    thexweight = 1.0
                newsample.weight=math.exp(newweight)*thexweight
                totweight += newsample.weight
                self.samples.append(newsample)
                    
            number_of_reset_attempts += 1



        if False and verb:
            print 100*"!","new samples:"
            printHeavySamples(self.samples,0.001)

        avgstate=sampleMean(self.samples)
        self.deflection_avg=NP.dot(avgstate.tau-avgstate.f,avgstate.tau-avgstate.f)
        self.x_avg=avgstate.x


        #prune POMCP tree if we are doing pomcp
        if self.use_pomcp:
            #possibly allow the user the explore the POMCP tree
            if self.pomcp_doInteractive:
                askq=raw_input("POMCP about to prune  - do you want to interactively explore the tree? (enter means no, anything else is yes):")
                if not askq == "":
                    self.pomcp_agent.POMCP_interactiveExplorePlanTree()
            #prune the tree
            if self.verbose:
                print 100*"~~"
                print aab,paab,observ,xobserv
            
            self.pomcp_pruneTree((aab,paab),(observ,xobserv))

        return avgstate



    #returns the propositional action to take in a given state
    #this can be overloaded in a subclass
    #for this base class, it is always the same
    def get_prop_action(self,state,rollout=False,samplenum=0):
        return 0


    #and is used in XXXYYYZZZ methods
    #it returns a default (null) action that is used when there is a "client" turn
    #using 0 as the null propositional action is problematic and should be fixed
    def get_null_action(self):
        #return ([0.0,0.0,0.0],0)
        return ([],0)

    #this is the mean of the prediction over F using only \psi (affect control principle)
    #but not using the inertia
    #if client turn, returns null action
    def get_default_action_noinertia(self,state):
        if state.get_turn()=="agent":
            tmpfv = getDefaultMeanF(fvars,tvars,self.iH,self.iC,state.tau,state.f,state.get_turn(),[],[])
            aab=map(lambda x: float(x), tmpfv[3:6])
            paab = self.get_prop_action(state)
        else:
            (aab,paab)=self.get_null_action()

        return (aab,paab)

    #get the predicted actions for the client, based on
    #what the agent would normally do in the same situation
    def get_default_predicted_action(self,state):
        #default ACT action selectors (see Bayesact paper)
        avgf=self.sample_next_f(state,self.nsamples_avgf)
        aab=map(lambda x: float(x), avgf[3:6])
        paab = self.get_prop_action(state)
        return (aab,paab)


    #what the agent should call to get a next action if using the default ACT choice
    #returns a null action if it is not the agent's turn
    #overload for subclass to define its own method for getting a default action -
    #this will be used if pomcp is not turned on to choose an action, and as the
    #first action chosen by POMCP when building a new branch of the planning tree
    def get_default_action(self,state):
        if state.get_turn()=="agent":
            (aab,paab)=self.get_default_predicted_action(state)
        else:
            (aab,paab)=self.get_null_action()

        return (aab,paab)

    #the greedy action selection mechanism as described in the Bayesact paper
    def get_greedy_action(self,state,paab=None):
        if paab==None:
            paab = self.get_prop_action(state)
        #with some hard-coded constants
	N=50
	M=20
        fvsamples=[]
        max_rrew=0
        best_action=[]

        fVarSampler = FVarSampler(self.isiga, self.isigf, state.tau, state.f, \
                                  self, state.get_turn(), [], [])
        for x in range(N):
            fv = fVarSampler.sampleNewFVar()
            fv = fv.transpose()
            aab=map(lambda x: float(x), fv[3:6])
            #simulate next state given that action
            rrew=0.0
            for y in range(M):
                sample = self.sampleNext(fVarSampler, fvars, tvars, state, \
                                         aab, paab)
                #get reward of that -
                rrew += self.reward(sample)
            rrew = rrew/M
            if x==0 or rrew>max_rrew:
                max_rrew = rrew
                best_action=aab
        return (best_action,paab)



    #get the next action using either the default ACT action selection
    #mechanism from the single state input (usually this is the average of the current f-values
    #or using pomcp
    #this returns the affective action, followed by the propositional action
    def get_next_action(self,state,paab=None,exploreTree=False):

        #this does not need to be called if use_pomcp is set, but
        #we do this anyways to be able to see the default action
        if True or not self.use_pomcp:
            #if client turn, returns the default (null) actions
            (aab,paab)= self.get_default_action(state)
            if self.verbose:
                print "myopic version: ",aab, " paab: ",paab


        #other options:

        #aab = self.get_default_action_noinertia(state)
        #print "myopic version (no inertia): ",aab


        #(aab,paab) = self.get_greedy_action(state)
        #print "greedy action: ",aab, " has value ",bestrew

        if self.use_pomcp and not self.pomcp_agent==None:
            if self.verbose:
                print "calling pomcp search"
            (avalue,aab) = self.pomcp_agent.POMCP_search(self.samples,self)
            paab=aab[1]
            aab=aab[0]
            if self.verbose:
                print "pomcp version: ",aab, " ", paab, "  has value: ",avalue

            if exploreTree:
                askq=raw_input("POMCP finished  - do you want to interactively explore the tree? (enter means no, anything else is yes):")
                if not askq == "":
                    self.pomcp_agent.POMCP_interactiveExplorePlanTree()


        return (aab,paab)

    #compute the deflection from a the current set of weighted samples
    def compute_deflection(self):
        self.deflection=0
        sumw=0
        for sample in self.samples:
            self.deflection += sample.weight*NP.dot(sample.f-sample.tau,sample.f-sample.tau)
            sumw += sample.weight
        self.deflection /= sumw
        return self.deflection

    #compute the id-deflection of an EPA value for other identity given in theid (3D vector)
    def compute_id_deflection(self,theid):
        id_deflection=0
        sumw=0
        for sample in self.samples:
            id_deflection += sample.weight*NP.dot(sample.f[6:9]-theid,sample.f[6:9]-theid)
            sumw += sample.weight
        id_deflection /= sumw
        return id_deflection

    #get the average state from a weighted set of samples
    def getAverageState(self):
        avgstate=sampleMean(self.samples)
        return avgstate
        
    def get_behaviour_label(self,fb):
        bid=findNearestBehaviour(fb,self.fbehaviours_agent)
        return bid

    def get_avg_ids(self,f):
        aid=findNearestIdentity(map(lambda x: x[1],filter(lambda x: x[0][1]=="a",zip(fvars,f))),self.fidentities_agent)
        cid=findNearestIdentity(map(lambda x: x[1],filter(lambda x: x[0][1]=="c",zip(fvars,f))),self.fidentities_client)
        return (aid,cid)

    #find identities across all samples
    def get_all_ids(self,subsetSize=-1):
        if self.verbose:
            print "extracting ids from samples ..."

        #if this is present, it means we want to this only for a random sample of subsetSize, to speed things up
        if subsetSize>0:
            subsetSamples = random.sample(set(self.samples),subsetSize)
        else:
            subsetSamples = self.samples
        agent_ids=map(lambda z: findNearestIdentity(map(lambda x: x[1],filter(lambda y: y[0][1]=="a",zip(fvars,z.f))),self.fidentities_agent),subsetSamples)
        client_ids=map(lambda z: findNearestIdentity(map(lambda x: x[1],filter(lambda y: y[0][1]=="c",zip(fvars,z.f))),self.fidentities_client),subsetSamples)

        cnt_ag={}
        for aid in agent_ids:
            if cnt_ag.has_key(aid):
                cnt_ag[aid] += 1
            else:
                cnt_ag[aid] = 1
        cnt_cl={}
        for cid in client_ids:
            if cnt_cl.has_key(cid):
                cnt_cl[cid] += 1
            else:
                cnt_cl[cid] = 1

        cnt_ags = sorted(cnt_ag.items(), key=lambda x: x[1],reverse=True)
        cnt_cls=sorted(cnt_cl.items(), key=lambda x: x[1],reverse=True)
        return (cnt_ags,cnt_cls)



    #----------------------------------------------------------------------------------
    #POMCP functions
    #----------------------------------------------------------------------------------
    # actions and observations are defined in the class that uses POMCP, so here
    # however, they are currently defined only by the way in which they are used by these functions
    # so actions are produced by oracle, and observations are produced by propagateSample

    #these functions are used by the POMCP class to call the agent and get a sample
    #this is the "blackbox simulator" used in POMCP
    def propagateSample(self,state,action):
        #continuous component is in the first member of the action tuple - this is encoded here in this class, not in pomcp class
        aab = action[0]

        #propositional action is the second member of the action tuple
        paab = action[1]

        #if agent turn, observ is ignored - but not in POMCP so this is awkward
        observ=NP.array([])

        #if client turn, we need to generate observ, and aab is ignored
        #very confusing - why was this here?  I have commented it out Feb 4, 2015
        #while working on PD-agents, but this seems quite suspect.  
        #seems that the observ on client turn should be drawn from the actual sample
        #(newsample.f) below  - so this is what I am doing now.
        if False and state.get_turn()=="client":
            #sample what client would do next
            fsampler = FVarSampler(self.isiga, self.isigf_unconstrained_b, \
                                   state.tau, state.f, self, "client")
            fsample = fsampler.sampleNewFVar()

            #sample an observation from fsample
            observ=self.sampleObservation(fvars,fsample,self.gamma_value_pomcp)

        #sample a next state
        fVarSampler = FVarSampler(self.isiga, self.isigf, state.tau, state.f, \
                                  self, state.get_turn(), aab, observ)
 
        newsample = self.sampleNext(fVarSampler, fvars, tvars, state, aab, paab)
        
        
        #here we should also get the observation on client turn
        if state.get_turn()=="client":
            #sample an observation from fsample
            observ=self.sampleObservation(fvars,newsample.f,self.gamma_value_pomcp)
        
        
        xobserv=self.sampleXObservation(newsample)


        #print 20*"xobserv"
        #print newsample.f[3:6],newsample.x,observ,xobserv
        


        #print 100*"~"
        #print "newsample is :"
        #newsample.print_val()
        #print "observ is: "
        #print observ
        #print "x observ is:"
        #print xobserv
        #print 100*"~"

        newreward = self.reward(state,(aab,paab))
        #turn is given as a noise-free observation here embedded in x
        return (newsample,(observ,xobserv),newreward)

    #draw a sample from a normal distribution with mean given by the middle three
    #values in f (using the f-variables in fvars) and with uniform diagonal covariance
    #with std. deviation gamma
    def sampleObservation(self,fvars,f,gamma):
        fb = getbvars(fvars,f)
        fbo=map(lambda fv: NP.random.normal(fv,gamma), fb)
        return fbo


    #rollout is a boolean flag  - if we are doing a rollout, then
    #this sample function computes and returns the rollout policy
    #samplenum is the number of times the oracle has been called at the node in the pomcp tree
    #this can be used to make a schedule for oracle actions
    def oracle(self,state,rollout=False,samplenum=None):
        #continuous component is just a 3D EPA vector which is the null value on client turn
        #the discrete component of the action is class dependent
        #here, if it is the client's turn, we simply use the null action for both
        #and in any case, the propositional action is always the same (0 = null)
        (a,paab)  = self.get_null_action()

        if state.get_turn()=="agent":
            paab = self.get_prop_action(state,rollout,samplenum)

            if not self.fixed_policy==None:
                return (self.fixed_policy[paab],paab)
            #the first sample returned is the default (ACT) action
            if not rollout and samplenum==0:
                (a,paab)=self.get_default_action(state)
            else:
                #afterwards, it is a random selection
                #and paab is always 0 anyways
                #isiga_pomcp is much smaller than isiga
                usesiga=self.isiga_pomcp
                #when doing a rollout, want to cast a wider net?
                if rollout:
                    usesiga=self.isiga
                fsampler = FVarSampler(usesiga, self.isigf_unconstrained_b, \
                                       state.tau, state.f, self, \
                                       state.get_turn())
                tmpfv = fsampler.sampleNewFVar()

                a = map(lambda x: float(x), tmpfv[3:6])
        return (a,paab)


    #prune the POMCP tree - action is a tuple (aab,paab)
    def pomcp_pruneTree(self,action,observ):
        (besta,besto,bestd) = self.pomcp_agent.POMCP_pruneTree(action,observ,self)
        if self.verbose:
            print "best a: ",besta
            print "best o: ",besto
            print "best d: ",bestd
        if (besta<0 or besto<0):
            #something went wrong, we need to reset the tree
            if self.verbose:
                print "something wrong - need to reset tree"
            self.initialise_pomcp(self.samples)

    #is true if obs1 and obs2 match to within resolv
    #observations are a tuple, with first component is the observation of behaviour (non-zero on client turn)
    #second component is an array of x values.
    #this must be overloaded in any subclass
    def observationMatch(self,obs1,obs2,resolv):
        return ( obs1[1]==obs2[1] and math.sqrt(raw_dist(obs2[0],obs1[0])) < resolv)

    #returns the closest observation match to obs in obsSet
    #this needs to overloaded in any subclass, as the comparison depends on the structure of the observation vector
    def bestObservationMatch(self,obs,obsSet,obsSetData):
        firstone=True
        bestdist=-1
        besto=-1
        for oindex in range(len(obsSet)):
            o=obsSet[oindex]
            if obs[1]==o[1]:
                odist = math.sqrt(raw_dist(obs[0],o[0]))
                if firstone or odist<bestdist:
                    firstone=False
                    besto=oindex
                    bestdist=odist
        return (besto,bestdist)

    #check to see if action is in actionSet (to within actres resolution)
    def hasAction(self,actionSet,action,actres):
        for a1 in actionSet:
            if a1[1]==action[1] and math.sqrt(raw_dist(action[0],a1[0])) < actres:
                return True
        return False


    #distance bewteen two actions - None means infinite
    #if == is not defined for the propositional action, then this must be overloaded in a subclass
    #must be overloaded in any subclass
    def actionDist(self,a1,a2):
        if a1[1]==a2[1]:
            return math.sqrt(raw_dist(a1[0],a2[0]))
        else:
            return -1

    #must be overloaded in any subclass
    def observationDist(self,obs1,obs2):
        if obs1[1] == obs2[1]:
            return math.sqrt(raw_dist(obs1[0],obs2[0]))
        else:
            return -2



    #an obsSet is a combination of two elements -
    #First, a N-d continuous vector and
    #Second, the the propositional observation,
    #which is actually class dependent
    #so here we depend on + and /scalar being defined for the observation arrays
    #needs to be generalised to it takes any observation and can be moved into pomcp.py perhaps
    #do this by making "x" an object - done now - this can be moved
    def getMeanObs(self,obsSet):
        if obsSet:
            avgobs=NP.array(obsSet[0][0])
            avgxobs=NP.array(obsSet[0][1])
            for o in obsSet[1:]:
                avgobs=avgobs+o[0]
                avgxobs = avgxobs+o[1]
            avgobs=avgobs/len(obsSet)
            avgxobs=avgxobs/len(obsSet)
            return (avgobs,avgxobs)
        else:
            return []


    #overload in subclass
    def reward(self,state,action=None):
        # a generic deflection-based reward
        fsample=state.f.transpose()
        reward = -1.0*NP.dot(state.f-state.tau,state.f-state.tau)
        return reward


    # The agent here is the agent that is propagating forward
    def sendSamplesToPlotter(self,samples,plotter,agent):
        # The plotter here is an instance of cPlotBayesactThread
        if (eTurn.learner == agent):
            plotter.m_LearnerFundamentalSamples = NP.array(map(lambda x: x.f, samples)).transpose().tolist()
            plotter.m_LearnerTauSamples = NP.array(map(lambda x: x.tau, samples)).transpose().tolist()
        #else eTurn.learner == agent
        else:
            plotter.m_SimulatorFundamentalSamples = NP.array(map(lambda x: x.f, samples)).transpose().tolist()
            plotter.m_SimulatorTauSamples = NP.array(map(lambda x: x.tau, samples)).transpose().tolist()
