
"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Simple command-line tool to find closest identities/behaviours to an EPA vector, or 
to find the EPA for an identity/behaviour
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

import re
import sys
import getopt
import numpy as NP
gender_list = ["male", "female"]
institution_list = ["lay","business","law","politics","academe","medicine","religion","family","sexual"]

sentiment_levels=2
smin=-4.3
smax=4.3
sincr=(smax-smin)/sentiment_levels

sentiment_indices=xrange(sentiment_levels+1)

sentiment_bins=[x*sincr+smin for x in list(xrange(sentiment_levels+1))]



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
        
        institution_list  = map(lambda i: bool(int(i)),list(re.sub(r"_","",y[7])))
        fsentiments_agent[y[0]]["institutions"]=institution_list

    fb.close()
    return fsentiments_agent

def getIdentity(iddic,theid):
    if iddic.has_key(theid):
        return NP.asarray([float(iddic[theid]["e"]),float(iddic[theid]["p"]),float(iddic[theid]["a"])])
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
    

#find nearest vector in dictionary fdict to epa-vector oval
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
    return [bestf,mind]


def main(argv):

    helpstring="Simple command-line EPA finder tool.\n Uage: python epafinder.py -e [<E,P,A>,label] -f <sentiment_file [fidentities.dat]> -g <gender [male]>\n\t argument to -e can be comma-separated e,p,a values or a label that is in the sentiment file. If the former, the closest label is returned. If the latter, the EPA value in the dictionary is returned."

    try:
        opts, args = getopt.getopt(argv[1:],"hf:g:e:",["help","f=","g=","e="])
    except getopt.GetoptError:
        print helpstring
        sys.exit(2)

    if opts==[]:
        print helpstring
        sys.exit(2)
    fmfname = "fidentities.dat"
    gender = "male"
    for opt, arg in opts:
        if opt == '-h':
            print helpstring
            sys.exit()
        elif opt == '-f':
            fmfname = arg
        elif opt == "-g":
            gender = arg
        elif opt == "-e":
            epa = arg

    try:
        epadict = readSentiments(fmfname,gender)
    except IOError:
        print "file not found: ",fmfname
        sys.exit(2)

    isLabel = False
    try:
        fepa = map(lambda x: float(x),epa.split(","))
    except ValueError:
        isLabel = True

    if isLabel:
        epaval = getIdentity(epadict,epa)
        if epaval == []:
            print "not found"
        else:
            print epaval
    else:
        [bestid,dist]= findNearestEPAVector(fepa,epadict)
        print "closest label:\t", bestid, "\n epa value:\t", getIdentity(epadict,bestid),"\n distance:\t",dist



if __name__ == "__main__":
    main(sys.argv)
        
