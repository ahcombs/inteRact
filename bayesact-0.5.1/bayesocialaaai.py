"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Bayesact of Self model
Author: Jesse Hoey  jhoey@cs.uwaterloo.ca   http://www.cs.uwaterloo.ca/~jhoey
July 2014
Use for research purposes only.
Please do not re-distribute without written permission from the author
Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by the Natural Sciences and Engineering Council of Canada (NSERC).
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
import wx
import wx.lib.dialogs
import wx.lib.scrolledpanel
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from bayesact import *


#some generic settings 
useEmployerAgent  = False 

#set both of these to False to do the initial learning of identities
identityFileBaseName = "data/fsamples_"
doPlotting = True

#does not work right onw - need to fix
doChatting = False

usePOMCP = False
#make this odd so the simulations end on client turn, so the next turn is agent when interactions take place.
identitySimulationT=21

#-----------------------------------------------------------------------------------------------------------------------------
#Learning agent stuff
#-----------------------------------------------------------------------------------------------------------------------------
knownWords={}
wordCounts={}


fifname="fidentities.dat"              
fbfname="fbehaviours.dat"



#-----------------------------------------
#Specify the fundamental self sentiment as
#a distribution over labels
#number of samples for the fundamental and situational self-sentiments
n_fsb=500
#agent_fsb gives the distribution over identities as a mixture of Gaussians
#each item of the list is a 3-tuple:  weight, identity label OR EPA vector, std. dev of the Gaussian
#agent_fsb = [[0.5,"female",0.05],[0.5,"employer",0.05]]
agent_fsb = [[0.5,"daughter",0.1],[0.5,"employer",0.1]]
#agent_fsb = [[0.4,"rapist",0.001],[0.4,"priest",0.001],[0.2,[2.5,2.5,1.0],0.05]]

#-----------------------------------------
#Specify people known initially - must be labels only
#ids of clients
#client_ids=["husband","employee"]
client_ids=["mother","employee"]
#client_ids=["victim","Catholic","same"]
#names so we can more easily keep track
client_names = ["mum","susan"]
#client_names = ["vicky","cathy","sam"]
#genders of clients
client_genders = ["female","female"]
#client_genders = ["female","female","male"]
#identity that each client maintains for the agent
#client_agent_ids = ["female","employer"]
client_agent_ids = ["daughter","employer"]
#OLD
#client_agent_ids = ["rapist","priest","father"]
## NEW VERSION for Kim's simulations
#client_agent_ids = ["thief","saint","politician"]

#use a stranger id as well as the ones above.
#the agent has a dispersed identity for the stranger, but then learns
#this identity (and the corresponding self sentiment) once it chooses to interact with her
#a new stranger is always added once the agent interacts with the stranger
#(so there is always another stranger that can be interacted with)
#IMPORTANT TO REMEMBER: these "strangers" are NOT the identity of "stranger" from the ACT database
#the actual identity of the stranger is chosen randomly when the agent chooses to interact with her
useStranger = True


#variances of initial ids for agent's model of client \beta_c^0
#and for simulated client's model of self  \beta_a^0 and \beta_c^0
client_variance= 0.1   #was 0.001
#can set these to be different for each item of client_ids
client_variances=len(client_ids)*[client_variance]

#variances for client's model of agent \beta_a^0 and \beta_c^0
client_agent_variance= 0.1   #was 0.001
#can set these to be different for each item of client_agent_ids
client_agent_variances = len(client_agent_ids)*[client_agent_variance]
#e.g. like this:
client_agent_variances = [0.1,0.5,0.05]

#NUMBER OF SAMPLES TO USE for the Bayesact simulations
#this needs to be rather high becauseo of interactions with the stranger, 
#but possibly it would be better to adjust some of the parameters (like gamma)
#for these interactions instead? Even make these parameters dynamically adjusting? 
n_samples=2000

#----------------------------------------------------------------------
# properties of agent, parameters
#----------------------------------------------------------------------

agent_gender = "female"

#how quickly self-sentiments update (learning rates)
eta = 0.75
eta_s = 0.5
eta_f = 0.95




#----------------------------------------------------------------------
# helper functions
#----------------------------------------------------------------------

#David Huard wrote this - found it on scipy.org
#compare two beliefs using kldivergence
def kldivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.    
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    
    References
    ----------
    Perez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree
    
    # Check the dimensions are consistent
    x = NP.atleast_2d(x)
    y = NP.atleast_2d(y)
    
    n,d = x.shape
    m,dy = y.shape
    
    assert(d == dy)
    
    
    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)
    
    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]
    
    #print r
    #print s
    # There is a mistake in the paper. In Eq. 14, the right side  misses a negative sign
    # on the first term of the right hand side.
    return -NP.log(r/s).sum() * d / n + NP.log(m / (n - 1.))

def writeMatlab(fname,suffix,fsb,ssb,thealters,theclientalters,wmode='w'):
    #write out fsb and the initial f_a[], f_c[] arrays
    f = open(fname,wmode)
    if not fsb==[]: 
        f.write("fsb=[")
        for x in fsb:
            for y in x:
                f.write(str(y)+" ")
            f.write(";\n")
        f.write("];\n")

    if not ssb == []:
        for alter in ssb:
            f.write(alter+suffix+"= [\n")
            for x in ssb[alter]:
                for y in x:
                    f.write(str(y)+" ")
                f.write(";\n")
            f.write("];\n")


    if not thealters==[]:
        for alter in thealters:
            f.write(alter+suffix+"= [\n")
            for x in thealters[alter]:
                for y in x.f:
                    f.write(str(y)+" ")
                f.write(";\n")
            f.write("];\n")

    suffix='c'+suffix
    if not theclientalters==[]:
        for alter in theclientalters:
            f.write(alter+suffix+"= [\n")
            for x in theclientalters[alter]:
                for y in x.f:
                    f.write(str(y)+" ")
                f.write(";\n")
            f.write("];\n")

    f.close()


#----------------------------------------------------------------------
# GUI widgets
#----------------------------------------------------------------------
class p1(wx.Panel):
    def __init__(self,parent):
        wx.Panel.__init__(self, parent)
        self.figure = plt.figure(figsize=(4.2, 4), dpi=80)
        plt.subplots_adjust(left=0.15,right=0.9,top=0.9,bottom=0.1)
        self.canvas = FigureCanvas(self,-1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Hide()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("e")
        self.ax.set_ylabel("p")
        self.ax.set_ybound(lower=-4.0, upper=+4.0)
        self.ax.set_xbound(lower=-4.0, upper=+4.0)
        self.ax.hold(False)

    def setHold(self,newhold):
        self.ax.hold(newhold)

    def plot(self,datax,datay,systring,lalpha=1.0):
        self.ax.plot(datax,datay,systring,alpha=lalpha)
        self.ax.set_ybound(lower=-4.0, upper=+4.0)
        self.ax.set_xbound(lower=-4.0, upper=+4.0)
        self.ax.set_xlabel("e")
        self.ax.set_ylabel("p")
        self.canvas.draw()    

    def plotId(self,theidepa,theid):
        #self.ax.text(theidepa[0],theidepa[1],"*")
        self.ax.plot(theidepa[0],theidepa[1],"y*",markersize=20)
        self.ax.set_ybound(lower=-4.0, upper=+4.0)
        self.ax.set_xbound(lower=-4.0, upper=+4.0)
        self.ax.set_xlabel("e")
        self.ax.set_ylabel("p")
        self.canvas.draw()
    


class IndividualChat(wx.Frame):
    def __init__(self,parent,title,name):
        wx.Frame.__init__(self,parent,title=title,size=(650,400), style=wx.FRAME_FLOAT_ON_PARENT | wx.MINIMIZE_BOX|wx.SYSTEM_MENU|
                          wx.CAPTION|wx.CLOSE_BOX|wx.CLIP_CHILDREN)

        self.name = name
        self.idplot = p1(self)
        #identity plot
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer = wx.BoxSizer(wx.VERTICAL)
        #for the chat log
        self.text = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.ctrl = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER, size=(300, 25))
        sizer1.Add(self.idplot, 5, wx.EXPAND)
        sizer1.Add(self.text, 5, wx.EXPAND)

        sizer.Add(sizer1, 5, wx.EXPAND)
        sizer.Add(self.ctrl, 0, wx.EXPAND)
        self.SetSizer(sizer)
        self.ctrl.Bind(wx.EVT_TEXT_ENTER, self.updateEvent)
        self.parentframe = parent
        self.mesez = True   #who has the turn ... yuck must get this fixed somehow
        
    def updateEvent(self, evt):
        self.update(self.ctrl.GetValue())

    #takes in the kldivergence.  So larger kl divergence should be more greyed out
    def setChatBackground(self,inauthenticity):
        cval = int(min(1,max(0,inauthenticity/2.0))*255)  #some arbitrary numbers in here - max kldivergence is 5? 
        self.ctrl.SetBackgroundColour((cval,cval,cval))

    def update(self,theChat):
        self.parentframe.theChat = theChat
        #print 100*"6",self.mesez
        if self.mesez:
            whosez = "\n you said: "
        else:
            whosez = "\n "+self.name+" says: "
        self.text.SetValue(self.text.GetValue()+whosez+theChat)
        self.ctrl.SetValue("")
        if self.mesez:
            self.parentframe.selectedBestAlter = self.name
            #print 100*"6","setting to SELECT"
            self.parentframe.state = "SELECT"
            self.mesez = False
        else:
            #this is only because I'm doing this f**ed up thing about the turns where 
            #I'm forced to do two turns each time
            #print 100*"6","setting to SELECT"
            self.parentframe.state = "UPDATE"   
            self.mesez = True

    def setClientChat(self, epaval):
        #self.mesez = False
        thechat = "#"+reduce(lambda x,y: x+"{0:.2f}".format(y)+",",epaval,"").strip(",")
        thechat = findNearestBehaviour(epaval,fbehaviours_agent) + "  "+ thechat
        self.ctrl.SetValue("  "+thechat)
        self.ctrl.SetInsertionPoint(0)
        
    def plotIdentities(self, f):
        self.idplot.plot(map(lambda x: x.f[6], f),map(lambda x: x.f[7],f),"ro")
        self.idplot.setHold(True)
        self.idplot.plot(map(lambda x: x.f[0], f),map(lambda x: x.f[1],f),"bo",lalpha=0.5)
        self.idplot.setHold(False)

        
        
        
    

class SelectIdentityFrame(wx.Dialog):
    def __init__(self,parent,title,fifname,instlist):
        wx.Dialog.__init__(self,parent,title=title,size=(480,400), style=wx.MINIMIZE_BOX|wx.SYSTEM_MENU|
                           wx.CAPTION|wx.CLOSE_BOX|wx.CLIP_CHILDREN)

        self.p1 = wx.Panel(self,style=wx.SUNKEN_BORDER)
        self.quitbutt = wx.Button(self.p1,-1,"done",pos=(260,300))
        self.quitbutt.Bind(wx.EVT_BUTTON,self.quitfn)
        self.fifname = fifname
        self.instlist=instlist 
        self.selectedIdentity = None

        #first, read in the identities
        fp = open(self.fifname,"rU")
        oneid = fp.readline()
        self.allids={}
        while not oneid == '':
            theid=[re.sub(r"\s+","_",i.strip()) for i in oneid.split(",")]
            self.allids[theid[0]] = map(lambda i: bool(int(i)),list(re.sub(r"_","",theid[7])))
            oneid = fp.readline()
        fp.close()


        #allids is a dictionary with keys of identity labels and values
        #are 14 booleans in a list, with the first two showing if the id is relevant to "male" or "female" respectively,
        #the next 9 showing relevance to institutions in the self.idlist
        #note that the actual EPA values are left off - these are used in bayesact.py (when an identity is created
        #in initialise_samples) but there, the identities file is read again every time it is needed.  Clearly this could
        #be streamlined.

        
        #add the institutions to a checkbox list
        self.selected_institutions = {}
        yi=0
        for theinst in self.instlist:
            self.selected_institutions[theinst] = wx.CheckBox(self.p1,  -1, theinst, (260, 10+yi*20))
            self.selected_institutions[theinst].SetValue(True)
            self.selected_institutions[theinst].Bind(wx.EVT_CHECKBOX,self.filterfn)
            yi += 1

        #add the gender to a checkbox list
        self.gs = {}
        for thegender in ["male","female"]:
            self.gs[thegender] = wx.CheckBox(self.p1,  -1, thegender, (260, 30+yi*20))
            self.gs[thegender].SetValue(True)
            self.gs[thegender].Bind(wx.EVT_CHECKBOX,self.filterfn)
            yi += 1

        #add the identities to a scrollable panel
        self.scrolledPanel = wx.lib.scrolledpanel.ScrolledPanel(self.p1, size=(240,300),style=wx.SUNKEN_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.idchoices={}
        self.idchoiceids={}
        yi=0
        for theid in self.allids:
            if yi == 0:
                self.idchoices[theid] = wx.RadioButton(self.scrolledPanel,  -1, theid, (10,10+yi*20),style=wx.RB_GROUP)
            else:
                self.idchoices[theid] = wx.RadioButton(self.scrolledPanel,  -1, theid, (10,10+yi*20))
            self.idchoiceids[self.idchoices[theid].GetId()] = theid
            self.idchoices[theid].Bind(wx.EVT_RADIOBUTTON, self.idChoiceFn)
            self.idchoices[theid].SetValue(False)
            yi += 1
        #and one last one outside the panel for a random choice
        self.randomChoice = wx.CheckBox(self.p1,  -1, "random choice", (10,320))
        self.randomChoice.SetValue(True)
        self.randomChoice.Bind(wx.EVT_CHECKBOX,self.chooseRandomFn)

        for theidchoice in self.idchoices:
            if not theidchoice == "random":
                sizer.Add(self.idchoices[theidchoice],1,wx.ALL|wx.EXPAND,5)
        self.scrolledPanel.SetSizer(sizer)
        self.scrolledPanel.Layout()
        self.scrolledPanel.SetupScrolling(scroll_x=False)

        #default choice is a random one, so we can make this now
        self.randchoice = True
        self.chooseRandomIdentity()

        
    def checkInstitutions(self, theid):
        yi=2
        passed = False
        for theinst in self.instlist:
            passed = passed or (self.selected_institutions[theinst].GetValue() and self.allids[theid][yi])
            yi += 1
        return passed
        
    def getSelectedInstitutions(self,inst_list):
        losi = len(inst_list)*[False]
        i=0
        for inst in inst_list:
            losi[i] = self.selected_institutions[inst].GetValue()
            i += 1
        return losi
            
    def filterfn(self,event):

        for theid in self.allids:
            if (((self.gs["male"].GetValue() and self.allids[theid][0]) or 
                 (self.gs["female"].GetValue() and self.allids[theid][1])) and 
                self.checkInstitutions(theid)):
                self.idchoices[theid].Enable()
            else:
                self.idchoices[theid].Disable()
        self.scrolledPanel.Layout()
        #choose a new random id - 
        self.chooseRandomIdentity()

    def idChoiceFn(self,event):
        self.selectedIdentity = self.idchoiceids[event.GetId()]
        self.randomChoice.SetValue(False)
        self.randchoice = False

    
    def chooseRandomIdentity(self):
        #choose randomly from the set of non-filtered ids
        availableids=[]
        for theid in self.allids:
            if (((self.gs["male"].GetValue() and self.allids[theid][0]) or 
                 (self.gs["female"].GetValue() and self.allids[theid][1])) and 
                self.checkInstitutions(theid)):
                availableids.append(theid)
        if availableids == []:
            #a default of stranger - this is the ACT database identity of stranger
            randid = "stranger"
            randgender = None
        else:
            randid = random.sample(availableids,1)[0]
            if self.allids[randid][0] and not self.allids[randid][1]:
                randgender = "male"
            elif self.allids[randid][1] and not self.allids[randid][0]:
                randgender = "female"
            else:
                #random choice of id
                if random.random() > 0.5:
                    randgender = "female"
                else:
                    randgender = "male"
                
            
        print "selected identity for stranger is currently: ",randid, " with gender: ",randgender
        self.selectedIdentity = randid
        self.selectedGender = randgender
        

    def chooseRandomFn(self,event):
        
        self.randchoice = self.randomChoice.GetValue()
        
        if self.randchoice:
            #rechoose another random identity
            self.chooseRandomIdentity()
            for theid in self.allids:
                self.idchoices[theid].SetValue(False)

    def quitfn(self,event):
        self.Close()



class SimulFrame(wx.Frame):
    def __init__(self,parent,title,T,known_client=True,run_once=False,run_auto=False, learn_avgs=[], simul_avgs=[], learn_turn="agent", simul_turn="client"):
        wx.Frame.__init__(self,parent,title=title,size=(700,600), style= wx.MINIMIZE_BOX|wx.SYSTEM_MENU| wx.FRAME_FLOAT_ON_PARENT | 
                  wx.CAPTION|wx.CLOSE_BOX|wx.CLIP_CHILDREN)
        self.bs = wx.BoxSizer(wx.HORIZONTAL)
        self.p1 = p1(self)
        self.p2 = p1(self)
        #the two plots
        self.bs.Add(self.p1,1,wx.EXPAND | wx.ALL, 1)
        self.bs.Add(self.p2,1,wx.EXPAND | wx.ALL, 1)
        self.b2 = wx.BoxSizer(wx.VERTICAL)
        self.p3 = wx.Panel(self,style=wx.SUNKEN_BORDER)
        self.b2.Add(self.bs,2,wx.EXPAND | wx.ALL, 1)
        self.b2.Add(self.p3,1,wx.EXPAND | wx.ALL, 1)
        self.T = T
        self.knownClient=known_client
        self.runAuto=run_auto
        self.runOnce=run_once

        #a quit button
        self.quitbutt = wx.Button(self.p3,-1,"quit",pos=(550,10))
        self.quitbutt.Bind(wx.EVT_BUTTON,self.OnClose)

        #a button to show the current word dictionary
        self.showWordsbutt = wx.Button(self.p3,-1,"show words",pos=(550,40))
        self.showWordsbutt.Bind(wx.EVT_BUTTON,self.showWords)

        #a button to pop up the filtering window button
        self.filterbutt = wx.Button(self.p3,-1,"filter institutions",pos=(390,40))
        self.filterbutt.Bind(wx.EVT_BUTTON,self.filterfn)

        #a legend for the plots
        bmp = wx.Image("./gui/images/circleMagenta.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_fsbCircle = wx.StaticBitmap(self.p3, -1, bmp, pos=(400,70), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_fsbStaticText = wx.StaticText(self.p3, -1, "Fundamental Self Sentiment", pos=(430, 70))

        bmp = wx.Image("./gui/images/circleGreen.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_ssbCircle = wx.StaticBitmap(self.p3, -1, bmp, pos=(400,90), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_ssbStaticText = wx.StaticText(self.p3, -1, "Situational Self Sentiment", pos=(430, 90))

        bmp = wx.Image("./gui/images/circleBlue.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_faCircle = wx.StaticBitmap(self.p3, -1, bmp, pos=(200,70), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_faStaticText = wx.StaticText(self.p3, -1, "Self Identity (f_a)", pos=(230, 70))

        bmp = wx.Image("./gui/images/circleRed.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_fcCircle = wx.StaticBitmap(self.p3, -1, bmp, pos=(200,90), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_fcStaticText = wx.StaticText(self.p3, -1, "Other Identity (f_c)", pos=(230, 90))

        bmp = wx.Image("./gui/images/goldStar.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_alterIdCircle = wx.StaticBitmap(self.p3, -1, bmp, pos=(200,110), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_alterIdStaticText = wx.StaticText(self.p3, -1, "True alter identity", pos=(230, 110))


        #a scrolling text frame for the log of actions taken (if not chatting interface only)
        if not doChatting:
            self.logtext = wx.TextCtrl(self.p3, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(700,100), pos=(10,150))
            self.logtext.SetInsertionPoint(0)
            self.bestNextAlterText = wx.StaticText(self.p3, label="best next alter: ", pos=(10,110))
            self.bestNextAlterName = wx.StaticText(self.p3, label="", pos=(110,110))


        #a start/stop button
        self.stepbutt = wx.Button(self.p3,-1,"step",pos=(290,10))
        self.stepbutt.Bind(wx.EVT_BUTTON,self.stepfn)

        #a run automatically button
        self.runbutt = wx.Button(self.p3,-1,"run auto",pos=(390,10))
        self.runbutt.Bind(wx.EVT_BUTTON,self.runfn)

        wx.StaticText(self.p3, -1, "Currently known alters:", (10, 40))
        self.alterChoices = wx.Choice(self.p3, -1, (10, 60), choices = [])
        self.alterChoices.Bind(wx.EVT_CHOICE, self.choicefn)

        #
        self.bestAlterlabel = wx.StaticText(self.p3, label="current alter :",pos=(10,10))
        self.bestAlterName = wx.StaticText(self.p3, label="",pos=(100,10))

        
        #self.sp.SplitVertically(self.p1,self.p2,650)
        self.SetSizer(self.b2)

        if not self.runOnce:
            if self.runAuto:
                self.state = 'UPDATE'
            else:
                self.state = 'STOPPED'
        else:
            self.runAuto = True
            self.state = 'UPDATE'
            self.updateCount = 0
            self.learn_turn=learn_turn
            self.simul_turn=simul_turn
            self.learn_avgs = learn_avgs
            self.simul_avgs = simul_avgs

        
        self.selectedBestAlter = None

        wx.EVT_IDLE(self, self.OnIdle) 
        wx.EVT_CLOSE(self, self.OnClose) 

    def choicefn(self, event):
        print('Alter Choice: %s\n' % event.GetString())
        self.selectedBestAlter = self.alterChoices.GetString(self.alterChoices.GetSelection())
        print "selected best alter: ",self.selectedBestAlter
        if self.selectedBestAlter == "stranger":
            self.idSelFrame.ShowModal()


    def filterfn(self, event):
        self.idSelFrame.ShowModal()

    def OnClose(self,event):
        self.Destroy()

    def showWords(self,event):
        printKnownWords()

    def stepfn(self,event):
        if self.state == 'STOPPED':
            self.state = 'SELECT'
        else:
            self.state = 'STOPPED'

    def runfn(self,event):
        if self.runAuto:
            self.runAuto = False
            self.runbutt.SetLabel("run auto")
            self.stepbutt.Show()
            self.state = 'STOPPED'
        else:
            self.runAuto = True
            self.runbutt.SetLabel("run step")
            self.stepbutt.Hide()
            self.state = 'SELECT'
            

    def setUpChatFrame(self,alter):
        self.chatFrame[alter] = IndividualChat(self,"Chat with "+alter,alter)
        self.chatFrame[alter].Show()
        
        learn_aab = self.me.bestNextActions[alter][0]
        simul_aab = self.me.bestNextActions[alter][2]
        
        self.chatFrame[alter].setClientChat(learn_aab)


    def setMe(self,theme):
        self.me = theme
        for alter in self.me.theAlters:
            self.alterChoices.Append(alter)
            
        #set up the identity selection frame, but don't add it or show it yet
        self.idSelFrame = SelectIdentityFrame(None,"Select Stranger Identity",self.me.fifname,self.me.institution_list)

        #set up a chat for each alter
        if doChatting:
            self.chatFrame = {}
            for alter in self.me.theAlters:
                self.setUpChatFrame(alter)

        (minkldiv,testbestAlter) = self.me.findBestAlter()
        print "last alter we interacted with was: ",self.me.bestAlter
        print "best alter for next time is: ",testbestAlter
        if not doChatting:
            self.bestNextAlterName.SetLabel(testbestAlter)
        #set this here? 
        self.selectedBestAlter = testbestAlter

    

    def plotSentiments(self,learnsamples,ssb,fsb):
        #plot the identities in one frame
        self.p1.plot(map(lambda x: x.f[6], learnsamples),map(lambda x: x.f[7],learnsamples),"ro")
        self.p1.setHold(True)
        self.p1.plot(map(lambda x: x.f[0], learnsamples),map(lambda x: x.f[1],learnsamples),"bo",lalpha=0.5)
        self.p1.setHold(False)
        #plot the situational and fundamental sentiment in the other
        self.p2.plot(map(lambda x: x[0], ssb),map(lambda x: x[1],ssb),"go")
        self.p2.setHold(True)
        self.p2.plot(map(lambda x: x[0], fsb),map(lambda x: x[1],fsb),"mo",lalpha=0.5)
        self.p2.setHold(False)

        #update the plot in the chat window as well
        if doChatting:
            self.chatFrame[self.me.bestAlter].plotIdentities(learnsamples)

    def plotTrueAlterIdentity(self, theidepa, theid):
        self.p1.setHold(True)
        self.p1.plotId(theidepa, theid)
        self.p1.setHold(False)



    def OnIdle(self,event):
        if self.state == 'STOPPED':
            if self.runOnce:
                #### ADD HERE UPDATE SSB SHIT
                self.simAlterResult =  (self.me.ssb,self.learn_avgs,self.simul_avgs,self.learn_turn,self.simul_turn)
                self.Close()
            return

        if self.state == 'SELECT':
            print "last alter we interacted with was: ",self.me.bestAlter
            self.me.bestAlter="stranger"

            #returns true if a stranger is chosen
            if self.me.selectInteractant(self.selectedBestAlter,self.idSelFrame.selectedIdentity,self.idSelFrame.selectedGender):
                #choose a new random identity in case we're running automatically
                self.idSelFrame.chooseRandomIdentity()
                self.alterChoices.Append(self.me.bestAlter)
                if doChatting:
                    #change the name on the chatframe
                    self.chatFrame[self.me.bestAlter] = self.chatFrame["stranger"]
                    self.chatFrame[self.me.bestAlter].name  = self.me.bestAlter
                    self.chatFrame[self.me.bestAlter].SetTitle("Chat with "+self.me.bestAlter)
                    #set up a new stranger
                    self.setUpChatFrame("stranger")
  

            #always becomes none
            self.selectedBestAlter = None
            self.alterChoices.SetSelection(self.alterChoices.FindString(self.me.bestAlter))

            self.bestAlterName.SetLabel(self.me.bestAlter)
            #initialise a new interaction
            #now, we are ready to carry out an interaction with bestAlter
            #the turns are fixed right now - we always take two turns and return to agent goes first  ... this needs work
            self.learn_turn="agent"
            self.simul_turn="client"
            (self.learn_avgs,self.simul_avgs) = self.me.initialise_alter(True)

            

            self.plotSentiments(self.me.learn_agent.samples,self.me.ssb,self.me.fsb)

            self.plotTrueAlterIdentity([self.me.learn_agent.fidentities_agent[self.me.theAlterIds[self.me.bestAlter]]["e"],
                                        self.me.learn_agent.fidentities_agent[self.me.theAlterIds[self.me.bestAlter]]["p"],
                                        self.me.learn_agent.fidentities_agent[self.me.theAlterIds[self.me.bestAlter]]["a"]],self.me.bestAlter)
                                       


            self.state = 'UPDATE'
            
            self.updateCount = 0
            return

        #print "self.state is : ",self.state
        if self.state == 'UPDATE':
        
            print 100*"-","iteration",self.updateCount
            print "about to simulate with alter: ",self.me.bestAlter, " learn turn: ",self.learn_turn

            (self.learn_aab,self.learn_paab,self.simul_aab,self.simul_paab) = self.me.bestNextActions[self.me.bestAlter]

            print "learn_aab: ",self.learn_aab
            print "learn_paab: ",self.learn_paab
            print "simul_aab: ",self.simul_aab
            print "simul_paab: ",self.simul_paab

            print 100*"a"

            the_selected_institutions = self.idSelFrame.getSelectedInstitutions(self.me.institution_list)
            print the_selected_institutions
            if self.learn_turn=="agent":
                thebeh = findNearestBehaviour(self.learn_aab,fbehaviours_agent,the_selected_institutions)
                print "action taken: ",thebeh
                print self.learn_aab
                theaction = "#"+reduce(lambda x,y: x+"{0:.2f}".format(y)+",",self.learn_aab,"").strip(",")

                if not doChatting:
                    self.logtext.AppendText("\n>you: "+thebeh+" "+theaction)
                    self.logtext.ScrollLines(-1)
            else:
                thebeh = findNearestBehaviour(self.simul_aab,fbehaviours_agent,the_selected_institutions)
                print "action taken: ",thebeh
                theaction = "#"+reduce(lambda x,y: x+"{0:.2f}".format(y)+",",self.simul_aab,"").strip(",")
                if not doChatting:
                    self.logtext.AppendText("\n>"+self.me.bestAlter+": "+thebeh+" "+theaction)
                    self.logtext.ScrollLines(-1)


            if doChatting:
                print "the chat was: ",self.theChat," <-----------> ",
                observ = processChat(self.me,fbehaviours_agent,self.theChat)
                print self.learn_aab
                print "observ was : ",observ
                #OK, here we actually replace the agent or client actions (which become the client/agent observations)
                #with the observation, which may be a chat, not an EPA value, so it needs to get converted to an EPA value
                #the observations for agent and client are both just the observ, since they can be either a chat, or an EPA value
                # see the AttributeError kludge in LearningAgent.evalSampleFvar that handles this type uncertainty.
                if self.learn_turn == "agent":
                    self.learn_aab = convertChatToEPA(observ)
                    self.simul_observ = self.learn_aab  # simulator is not a learningAgnet
                    self.learn_observ = []
                else:
                    self.simul_aab = convertChatToEPA(observ)
                    self.simul_observ = []
                    self.learn_observ = observ
            else:
                if self.learn_turn == "agent":
                    self.learn_observ = []
                    self.simul_observ = self.learn_aab
                else:
                    self.learn_observ = self.simul_aab
                    self.simul_observ = []
                #observ may be a chat (if there was no hashtag), or a EPA vector (if there was a hashtag with a label or EPA value)
            (new_learn_avgs,new_simul_avgs,new_learn_turn,new_simul_turn) = self.me.simulate_alter_onestep(self.learn_aab,self.learn_paab,self.learn_observ,self.simul_aab,self.simul_paab,self.simul_observ,True,self.learn_turn,self.simul_turn,self.learn_avgs,self.simul_avgs)
    
            if not self.runOnce:
                self.me.updateAlters()
                self.me.updateSelfSentiment(True)

            self.learn_avgs = new_learn_avgs
            self.simul_avgs = new_simul_avgs
            self.learn_turn = new_learn_turn
            self.simul_turn = new_simul_turn



            self.plotSentiments(self.me.learn_agent.samples,self.me.ssb,self.me.fsb)

            #self.plotTrueAlterIdentity([0,0,0],self.me.bestAlter)
            self.plotTrueAlterIdentity([self.me.learn_agent.fidentities_agent[self.me.theAlterIds[self.me.bestAlter]]["e"],
                                        self.me.learn_agent.fidentities_agent[self.me.theAlterIds[self.me.bestAlter]]["p"],
                                        self.me.learn_agent.fidentities_agent[self.me.theAlterIds[self.me.bestAlter]]["a"]],self.me.bestAlter)
                                       

            self.me.bestNextActions[self.me.bestAlter]=self.me.get_recommended_next_actions(True,self.learn_avgs,self.simul_avgs)

            print "got recommended actions for next time: ",self.me.bestNextActions[self.me.bestAlter]

            learn_aab = self.me.bestNextActions[self.me.bestAlter][0]
            simul_aab = self.me.bestNextActions[self.me.bestAlter][2]
            
            
            (minkldiv,testbestAlter) = self.me.findBestAlter()
            print 10*"^"
            if self.learn_turn == "agent":
                print "best alter for next time is: ",testbestAlter
                self.bestNextAlterName.SetLabel(testbestAlter)
                self.selectedBestAlter = testbestAlter
            
            if doChatting:
                #this is who is going to act next
                if self.learn_turn == "agent":
                    self.chatFrame[self.me.bestAlter].setClientChat(learn_aab)
                    self.state = 'STOPPED'
                    return

                if self.learn_turn == "client":
                    #self.chatFrame[self.me.bestAlter].setClientChat("#"+reduce(lambda x,y: x+"{0:.2f}".format(y)+",",simul_aab,"").strip(","))
                    theChat = "#"+reduce(lambda x,y: x+"{0:.2f}".format(y)+",",simul_aab,"").strip(",")
                    #ask user here to type something in the console for testing purposes
                    suggestedWords = convertEPAToChat(simul_aab,1.0,2)
                    clientChat = raw_input("say something (suggestion: "+suggestedWords+") : ")
                    theChat = clientChat + "  "+theChat
                    self.state = 'STOPPED'
                    self.chatFrame[self.me.bestAlter].update(theChat)
                

                    #update the chat window shades to show the inauthenticity
                    if doChatting:
                        (minkldiv,bestAlter) = self.me.findBestAlter(self.chatFrame,doChatting)
                    #Now, we want to call update on the chatFrame
                    #this will actually happen because of the real client sending a chat
                    return
            
            if self.runAuto:
                if self.updateCount >= (self.T-1):
                    if self.runOnce:
                        self.state = 'STOPPED'
                    else:
                        self.state = 'SELECT'
                else:
                    self.state = 'UPDATE'  #no change
                    self.updateCount += 1
            else:
                #have to do at least two steps
                if self.updateCount >= 1:
                    self.state = 'STOPPED'
                else:
                    self.state = 'UPDATE'  #no change
                    self.updateCount += 1
            

#----------------------------------------------------------------------
# Sample Initialisers
#----------------------------------------------------------------------


class EmployerSampleInitialiser(SampleInitialiser):
    def __init__(self,*args,**kwargs):
        super(EmployerSampleInitialiser, self).__init__(self,*args,**kwargs)


    #overridden from original class in bayesact.py
    def initialise_sample_x(self, initx):
        return [list(NP.random.multinomial(1,initx[0])).index(1), 
                list(NP.random.multinomial(1,initx[1])).index(1)]


#----------------------------------------------------------------------
# a special case of Agent
#----------------------------------------------------------------------

#a simple test agent that has one binary x-variable on top of the turn
#this x state could represent selling a product or something for the employer
#x will transition to 1  (product sold) only if the person has the identity of employer
#the reward function will be centered around selling the product
class EmployerAgent(Agent):
    def __init__(self,*args,**kwargs):

        super(EmployerAgent, self).__init__(self,*args,**kwargs)
        #x for the DiscreteTutoringAgent is a pair:
        #the first value is the turn (always this way for any Agent)
        #the second value
        #is the binary objective state
        #probability distribution over initial x
        self.px=kwargs.get("initx",[1.0,0.0])
        self.ppx = [[0.8,0.2],[1.0,0.0]]
        self.of=kwargs.get("of",[[0.99,0.01],[0.01,0.99]])

    def print_params(self):
        Agent.print_params(self)
        print "px: ",self.px

    #The only difference between an EmployerAgent and an Agent
    #is dynamics of x and reward
    def initialise_x(self,initx=None):
        if not initx:
            initpx=self.px
            initturn=State.turnnames.index("agent")
        else:
            initpx=initx[1]
            initturn=initx[0]
        initobjective = list(NP.random.multinomial(1,initpx)).index(1)
        return [initturn,initobjective]

    def sampleXObservation(self,state):
        if state.get_turn()=="agent":
            return [state.x[0],list(NP.random.multinomial(1,self.of[state.x[1]])).index(1)]
        else:
            return [state.x[0],0]


    #aab is the affective part of the action, paab is the propositional part
    def sampleXvar(self,f,tau,state,aab,paab):
        if state.get_turn()=="client":
            #compute difference between f and the employer id 
            #x moves towards goalx inversely proportionally to 
            #the deflection
            employerid = [1.48,1.93,0.74]
            D=NP.dot(f[0:3]-employerid,f[0:3]-employerid)
            #as deflection from employer grows, the distribtuion is more likely to favor staying the same or regressing, not progressing
            xppx=NP.asarray(self.ppx[state.x[1]])
            #current and all previous ones as wellget increased by deflection
            xppx[0] = xppx[0]+D   #totally arbitrary
            #renormalize
            xppx = xppx/float(xppx.sum())
            #sample from this to get the new x value
            new_x=list(NP.random.multinomial(1,xppx)).index(1)
            #print xppx
            #print new_x
            return [state.invert_turn(),new_x]
        else:
            return [state.invert_turn(),state.x[1]]
    
    #this must be in the class
    def evalSampleXvar(self,sample,xobs):
        #this nasty little hack is because I'm calling update(..) to update the agent
        #in tutorgui2.py, which makes a default assumption that on agent turn, no observations
        #are received. 
        # ?? this true here in employeragent?
        #if xobs==[] or not turn=="client":
        #    return 1.0
        #only when it was the client turn (so it is now agent turn) do we use the observation function
        if sample.get_turn()=="agent":
            if xobs[0]==sample.x[0]:
                return self.of[sample.x[1]][xobs[1]]
            else: 
                return 0.0
        else:
            return 1.0   #ignored x observation

    def reward(self,sample,action=None):
        # a generic deflection-based reward
        fsample=sample.f.transpose()
        freward = -1.0*NP.dot(sample.f-sample.tau,sample.f-sample.tau)
        # a state-based reward that favors x=1
        xreward  = 0.0
        if sample.x[1] == 1:
            xreward += 100.0

        return freward+xreward


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

#the only purpose of this subclass of Agent is to 
#override sampleObservation such that POMCP works properly when planning how to interact 
#with a Stranger (who, in simulation, always gives [] as the observation \omega_f
if useEmployerAgent:
    print 100*"!"
    print 100*"!","CHANGE superclass of StrangerAgent to be EmployerAgent!!"
    print 100*"!"

class StrangerAgent(Agent):
    def __init__(self,*args,**kwargs):

        super(StrangerAgent, self).__init__(self,*args,**kwargs)

    #return a null sample as the observation
    def sampleObservation(self,fvars,f,gamma):
        return []

#------------------------------------------------------------------------------
# the learning agent class and related functions - must override evalSampleFvar
#------------------------------------------------------------------------------

def addWordToKnownWords(word,fbs,weight):
    if not word in knownWords:
        wordCounts[word] = 0
        knownWords[word] = []
    wordCounts[word] += 1
    knownWords[word].append((NP.array(fbs),weight))

def addChatToKnownWords(chat,fbs,weight):
    for word in chat.split():
        addWordToKnownWords(word,fbs,weight)

def getWeight(fb,sentence,h):
    sum_weight=0
    for word in sentence.split():
        sum_weight += math.log(getWeightWord(fb,word,h))
    return sum_weight

def getWeightWord(fb,word,h):
    #compute the kernel estimate of the density given the word
    weight=0.0
    for (fbi,wgt) in knownWords[word]:
        weight += wgt*math.exp(-1.0*raw_dist(fb,fbi)/h)
    return weight

def getWeightedMean(fsample):
    meanf = NP.array([0.0,0.0,0.0])
    smwgt = 0.0
    for f in fsample:
        meanf = meanf + f[1]*f[0]
        smwgt += f[1]
    return meanf/smwgt

def convertChatToEPA(chat):
    if type(chat) == type([]):
        #its already an EPA
        return chat
    for tries in range(10):
        fsample=[]
        for word in chat.split():
            if word in knownWords:
                fsample.append(random.sample(knownWords[word],1)[0])
        if fsample==[]:
            print 100*"!","returning default f for chat"
            return [0.5,0.5,0.5]   # some default value
        dist=0
        print fsample
        for i in range(len(fsample)):
            for fj in fsample[i+1:]:
                fdiff = fsample[i][0]-fj[0]
                dist += NP.dot(fdiff,fdiff)
        
        if tries == 0 or dist < mindist:
            mindist = dist
            bestf = getWeightedMean(fsample)

    return bestf

def convertEPAToChat(epa,h,numwords):
    weightWord = {}
    for word in knownWords:
        weightWord[word] = getWeightWord(epa,word,h)
    sortedWords = sorted(weightWord.items(), key=lambda x: x[1])
    chat=  reduce(lambda x,y: x+y[0]+" ",sortedWords[-numwords:],"").strip()
    return chat

def printKnownWords():
    for word in knownWords:
        print "word is :",word
        #print fsampleMean(knownWords[word])
        print knownWords[word]        

def is_array_of_floats(possarray):
    try:
        parr = [float(x) for x in possarray.split(',')]
    except ValueError:
        return False
    if len(parr)==3:
        return parr
    return False

#takes a behaviour dictionary and a chat
#if the chat contains a hashtag, process the word after the hashtag 
#it can be either a label from the dictionary fbeh, or a set of three floats,
#in either of these cases, return the EPA value (either the set of floats or 
#the looked up EPA value from the label).  Finally, the chat is then added to the
#dictionary with the EPA value as computed. 
#If the hashtag is not a valid label or EPA value, or if there is no hashtag, 
#the chat is returned as is, with spaces removed.
def processChat(theme,fbeh,cact):
    cchat=""
    behlabel=""
    for word in cact.split():
        if word[0]=='#':
            behlabel=word[1:]
        else:
            #possibly do other clean up of the chat here
            cchat = cchat+" "+word
                    
    #see if hashtag corresponds to a label in the dictionary
    if not behlabel=="":
        if re.sub(r"\s+","_",behlabel.strip()) in fbeh.keys():
            cact=re.sub(r"\s+","_",behlabel.strip()) 
            observ = map(lambda x: float (x), [fbeh[cact]["e"],fbeh[cact]["p"],fbeh[cact]["a"]])
        elif is_array_of_floats(behlabel):
            observ = [float(x) for x in behlabel.split(',')]
        else:
            #hashtag was not found, so ignore and continue as usual
            theme.addChat(cchat)
            return cchat
        #enter the chat into the dictionaries with a weight of 1  (the largest possible weight)
        addChatToKnownWords(cchat,observ,1)
        #add also based on current samples
        theme.addChat(cchat)
        #return the observ value (EPA value)
        return observ
    #else, return the chat
    theme.addChat(cchat)
    return cchat


class LearningAgent(Agent):
    def __init__(self,*args,**kwargs):
        
        super(LearningAgent, self).__init__(self,*args,**kwargs)
        self.h = kwargs.get("kernel_scale",1.0)
        
    def initialise_x(self,initx):
        if NP.random.random() > 0.5:
            return [State.turnnames.index(initx)]
        else:
            return [State.turnnames.index(invert_turn(initx))]

    #evaluates a sample of Fvar'=state.f
    #the observ now is a word which will need to be looked up
    
    def evalSampleFvar(self,fvars,tdyn,state,ivar,ldenom,turn,observ):
        weight=0.0
        if (not observ==[]): # and turn=="client":
            fb=getbvars(fvars,state.f)
            try:
                #super inefficient because it checks if each word in the chat is known every time this is called (for every sample)
                #assume this will not get done if the line below (getWeight) throws the Attribute Exception, which
                #means that observ is actually a list of floats  - this seems really awkward
                weight += getWeight(fb,observ,self.h)
            except AttributeError:  #super lame
                #standard method
                dvo=NP.array(fb)-NP.array(observ)
                weight += self.normpdf(dvo,0.0,ivar,ldenom)
                
        else:
            weight=0.0
        return weight

    #overload in subclass
    def sampleXvar(self,f,tau,state,aab,paab=None):
        if NP.random.random() > 0.2:
            return [state.invert_turn()]
        else:
            return state.x


    def getFbSamples(self,fvars):
        fb=[]
        for state in self.samples:
            fb.append(getbvars(fvars,state.f))
        return fb

    def drawFbSample(self,fvars):
        newsample=drawSamples(self.samples,1)
        state = newsample[0]
        #(tmpfv,wgt,tmpH,tmpC)=sampleFvar(fvars,tvars,self.iH,self.iC,self.isiga,self.isigf_unconstrained_b,state.tau,state.f,state.get_turn(),[],[])
        fVarSampler = FVarSampler(self.isiga, self.isigf, state.tau, state.f, \
                                  self, state.get_turn(), [], [])
        fv = fVarSampler.sampleNewFVar()
        fv = fv.transpose()

        fb = getbvars(fvars,fv)
        return fb


    def addChat(self,fvars,observ):
        try:
            fb=self.drawFbSample(fvars)
            addChatToKnownWords(observ,fb,0.01)  #small weight here
        except AttributeError:  #lame
            return
        return

# aversion of learningagent for strangers
class StrangerLearningAgent(LearningAgent):
    def __init__(self,*args,**kwargs):

        super(StrangerLearningAgent, self).__init__(self,*args,**kwargs)

    #return a null sample as the observation
    def sampleObservation(self,fvars,f,gamma):
        return []



#----------------------------------------------------------------------
# the main Self class
#----------------------------------------------------------------------
    
class Self(object):
    #---------------------------------------------------------------------------
    #constructor
    #---------------------------------------------------------------------------
    def __init__(self,*args,**kwargs):
        #rate at which the situational self-sentiment persists over time
        #(1-eta_value) is the learning rate
        self.eta_value=kwargs.get("eta_value",0.75)
        self.etaf_value=kwargs.get("etaf_value",0.75)
        self.etas_value=kwargs.get("etas_value",0.75)
        
        self.N_fsb = kwargs.get("n_fsb",1000)
        self.N_ssb = kwargs.get("n_ssb",1000)

        self.num_samples = kwargs.get("n_samples",1000)

        #we may eventually want these to be dynamic and possibly different for different alters
        self.alpha = 0.1
        self.alpha_a = 0.1
        self.gamma = 0.1
        self.gamma_a = 0.1
        self.beta_a = 0.01 #0.01 works well

        self.beta_variance = 0.01

        self.env_noise=0.0

        #POMCP parameters 
        self.numcact=5
        #number of discrete (propositional) actions
        #this should be set according to the domain, and is 1 for this generic class
        #one discrete action really means no choice
        self.numdact=1
        #observation resolution when building pomcp plan tree
        self.obsres=2.0
        #action resolution when buildling pomcp plan tree
        self.actres=1.0
        #timeout used for POMCP
        self.timeout=2.0  #should be bigger, but for testing, we cut corners

        self.doPlotting = kwargs.get("do_plot",False)
        self.gender = kwargs.get("gender","female")
        self.fifname = kwargs.get("fifname","fidentities.dat")

        self.institution_list = ["lay","business","law","politics","academe","medicine","religion","family","sexual"]


        self.theAlters = {}
        self.theAlterIds = {}
        self.theClientAlters = {}
        self.theClientGenders = {}
        self.ssbClients={}
        self.bestNextActions={}

        self.bestAlter = None

    def setPlotting(self,doplot):
        self.doPlotting = doplot
    #compares two belief states using EMD measure
    def compareBeliefs(self,b1,b2):
        beliefDissimilarity = 0.0
        if len(b1)==0 or len(b2)==0:
            return

        #loop over dimensions
        biggestDiff = -1.0
        for d in range(len(b1[0])):
            b1d = sorted(map(lambda x: x[d],b1))
            b2d = sorted(map(lambda x: x[d],b2))
            j1=0
            j2=0
            for b in b1d:
                while j2 < len(b2d) and b2d[j2] < b:
                    j2 = j2 + 1
                j1 = j1 + 1 
                biggestDiff = max(biggestDiff,abs(j2-j1))
        return 1.0*biggestDiff/max(len(b1),len(b2))


    def startNewFriend(self,cid,cgen):

        numfriends = len(self.theAlters)
        sampleiniter = SampleInitialiser()
        if self.useEmployerAgent:
            initx0 = [[1.0,0.0],[1.0,0.0]]
            esampleiniter = EmployerSampleInitialiser()
        else:
            initx0 = [1.0,0.0]
            esampleiniter = SampleInitialiser()

        initx0c=[0.0,1.0]

        client_gender = cgen     #was "female" always - world of women

        print 20*"%"," id randomly selected is: ",cid, " with gender: ",client_gender
        
        cvar = 0.1

        name = cid+"_"+"new"+str(numfriends)

        #get default stats from fifname
        (msb,csb) = getIdentityStats(fifname,self.gender)
        
        
        #f_a and f_c for agent
        self.theAlters[name] = esampleiniter.initialise_samples(self.num_samples,self.fifname,self.gender,
                                                                self.agent_ids,[[1.0,msb,csb]],initx=initx0)

        self.theAlterIds[name]=cid
        
        #f_a and f_c for client simulations
        self.theClientAlters[name] = sampleiniter.initialise_samples(self.num_samples,self.fifname,client_gender,
                                                                        [[1.0,cid,cvar]],[[1.0,msb,csb]],initx=initx0c) 

        self.theClientGenders[name] = client_gender

        self.ssbClients[name] = copy.deepcopy(self.fsb)

        self.bestNextActions[name] = self.bestNextActions["stranger"]

        
        #set up this alter and find the first action to take for her
        #already been done in initialiseAllAlters
        #self.bestAlter = name
        #(learn_avgs,simul_avgs) = self.initialise_alter(True,"agent","client")
        #self.bestNextActions[self.bestAlter]=get_recommended_next_actions(self,True,learn_avgs,simul_avgs)

        return name

    def findBestAlter(self,chatFrame=[],colorChatFrames=False):
        minkldiv=0.0
        first_alter = True
        kldiv={}
        for alter in self.theAlters:
            #now, we combine the situational of two identities and see which one works bettter
            #combine ssb with ssb_clients[alter1] and see which one works best
            ssb_mixed = copy.deepcopy(self.ssb)
            for i in range(len(ssb_mixed)):
                if NP.random.random() > self.etas_value:
                    #replace with sample from ssb_clients[alter1]
                    ssb_mixed[i] = copy.deepcopy(self.ssbClients[alter][i])
            #add a bit of random noise so the KL divergence doesn't diverge
            ssb_mixed = map(lambda x: x + NP.random.random([3])*0.01,ssb_mixed)
            #compare that result to the fundamental 
            kldiv[alter] = kldivergence(ssb_mixed,self.fsb)
            print "interaction with ", alter," will have a divergence of: ",kldiv[alter]
            if first_alter or abs(kldiv[alter]) < minkldiv:
                bestalter=alter
                minkldiv = abs(kldiv[alter])
                first_alter=False
            if colorChatFrames:
                chatFrame[alter].setChatBackground(kldiv[alter])
        return (minkldiv,bestalter)

    def selectInteractant(self,selectedBestAlter,selectedIdentity,selectedGender):
        if selectedBestAlter==None:
            (minkldiv,self.bestAlter) = self.findBestAlter()
            print "best alter to interact with is ",self.bestAlter," with a kl divergence of ",minkldiv
        else:
            print "selected alter: ",selectedBestAlter
            self.bestAlter = selectedBestAlter


        if self.bestAlter=="stranger":
            #add new stranger
            print "add new stranger"
            #to do this, we need to "rename" the stranger to something else I guess
            #and then add a new "stranger" id
            #he's no longer a stranger now, so we need to copy over the default "unknown" identity here 
            #and find out who he is! However this will not really work in simulation, since we should select
            #some client alter from a list of possible strangers and instantiate this one to be him
            #*********we should do this by:
            #1. drawing a random sample from the list of identities
            #2. initialising an agent with f_a (thealter) = fsb, f_c (for agent) is "unknown", 
            #    and theclientalter with this identity (this is f_a for simul), and the simulator with f_c as "unknown" (or not?) 
            #3. then let the interaction actually figure out who this is - possibly little by little
            #I think this has been done now
            self.bestAlter = self.startNewFriend(selectedIdentity,selectedGender)

            return True
        return False  #not a stranger
        
    def updateAlters(self):
        self.theAlters[self.bestAlter] = self.learn_agent.samples
        self.theClientAlters[self.bestAlter] = self.simul_agent.samples

    def updateSelfSentiment(self,known_client):
        if known_client:
            ssb_mixed = copy.deepcopy(self.ssb) 
            for i in range(len(ssb_mixed)):
                if NP.random.random() > self.eta_value:
                    #replace with sample from self.ssb_clients[alter1]
                    ssb_mixed[i] = copy.deepcopy(random.sample(self.ssbClients[self.bestAlter],1)[0])
            #the new self-sentiment for the ssb_client!
            self.ssbClients[self.bestAlter] = ssb_mixed
        else:
            self.ssbClients[self.bestAlter] = copy.deepcopy(self.ssb)
        
        #update fundamental self sentiment
        fsb_mixed = copy.deepcopy(self.fsb) 
        for i in range(len(ssb_mixed)):
            if NP.random.random() > self.etaf_value:
                #replace with sample from self.ssb
                fsb_mixed[i] = copy.deepcopy(random.sample(self.ssb,1)[0])
            #the new self-sentiment for the ssb_client!
            self.fsb = fsb_mixed


    def addChat(self,theChat):
        self.learn_agent.addChat(fvars,theChat)


    def initialiseAllAlters(self):
        for alter in self.theAlters:
            known_client=True
            if alter == "stranger":
                known_client=False
            self.bestAlter = alter
            (learn_avgs,simul_avgs) = self.initialise_alter(known_client)
            #this can't happen here because the POMCP tree is not saved !!  Yikes, we also need to save this if using POMCP
            self.bestNextActions[alter]=self.get_recommended_next_actions(known_client,learn_avgs,simul_avgs)
                
    def initialise_alter(self,known_client=False):

        alter = self.theAlters[self.bestAlter]
        clientAlter = self.theClientAlters[self.bestAlter]
        client_gender = self.theClientGenders[self.bestAlter]

        #look in Self constructor now for setting these parameters
        alpha = self.alpha
        alpha_a = self.alpha_a
        gamma = self.gamma
        gamma_a = self.gamma_a
        beta_a = self.beta_a

        beta_variance = self.beta_variance

        #this roughening noise disrupts things siginficantly, but 
        #probably not for the worse
        roughening_noise=self.num_samples**(-1.0/3.0)
        
        #otherwise, to get really obvious results, use no roughening noise, but 
        #then the identities are not as flexible.
        #roughening_noise=0.0
        #roughening_noise=0.05



        if known_client:
            agent_knowledge=2
            client_knowledge=2
            
            #world of women - fix this

            #the simulator agent - this is the actual client, possibly with an unknown identity to the agent
            #the agent_gender for the simulator is the gender of the client (#confusing)
            self.simul_agent=Agent(N=self.num_samples,alpha_value=alpha,gamma_value=gamma,beta_value_agent=beta_variance,beta_value_client=beta_variance,agent_rough=roughening_noise,client_rough=roughening_noise,identities_file=self.fifname,agent_gender=client_gender)
            simul_avgs = self.simul_agent.set_samples(clientAlter)

            #the learner agent  - the agent that represents the self's enactment of an identity
            if useEmployerAgent:
                #possibly as an "employerAgent" who has a different reward function
                self.learn_agent=EmployerAgent(N=self.num_samples,alpha_value=alpha_a,gamma_value=gamma_a,beta_value_agent=beta_a,beta_value_client=beta_variance,agent_rough=roughening_noise,client_rough=roughening_noise,identities_file=self.fifname,use_pomcp=usePOMCP,numcact=self.numcact,numdact=self.numdact,obsres=self.obsres,actres=self.actres,pomcp_timeout=self.timeout,alpha_pomcp=alpha_a,gamma_pomcp=gamma, agent_gender=self.gender)
            elif doChatting:
                #we are doing the chat interface, so the agent will learn EPA ratings for words
                self.learn_agent=LearningAgent(N=self.num_samples,alpha_value=alpha_a,gamma_value=gamma_a,beta_value_agent=beta_a,beta_value_client=beta_variance,agent_rough=roughening_noise,client_rough=roughening_noise,identities_file=self.fifname,use_pomcp=usePOMCP,numcact=self.numcact,numdact=self.numdact,obsres=self.obsres,actres=self.actres,pomcp_timeout=self.timeout,alpha_pomcp=alpha_a,gamma_pomcp=gamma, agent_gender=self.gender)
            else:
                #generic agent
                self.learn_agent=Agent(N=self.num_samples,alpha_value=alpha_a,gamma_value=gamma_a,beta_value_agent=beta_a,beta_value_client=beta_variance,agent_rough=roughening_noise,client_rough=roughening_noise,identities_file=self.fifname,use_pomcp=usePOMCP,numcact=self.numcact,numdact=self.numdact,obsres=self.obsres,actres=self.actres,pomcp_timeout=self.timeout,alpha_pomcp=alpha_a,gamma_pomcp=gamma, agent_gender=self.gender)
        else:
            agent_knowledge=0
            #a special kind of agent that works for POMCP when doing interactions with a stranger
            if doChatting:
                self.learn_agent=StrangerLearningAgent(N=self.num_samples,alpha_value=alpha_a,gamma_value=gamma_a,beta_value_agent=beta_a,beta_value_client=beta_variance,agent_rough=roughening_noise,client_rough=roughening_noise,identities_file=self.fifname,use_pomcp=usePOMCP,numcact=self.numcact,numdact=self.numdact,obsres=self.obsres,actres=self.actres,pomcp_timeout=self.timeout,alpha_pomcp=alpha_a,gamma_pomcp=gamma, agent_gender=self.gender)
            else:
                self.learn_agent=StrangerAgent(N=self.num_samples,alpha_value=alpha_a,gamma_value=gamma_a,beta_value_agent=beta_a,beta_value_client=beta_variance,agent_rough=roughening_noise,client_rough=roughening_noise,identities_file=self.fifname,use_pomcp=usePOMCP,numcact=self.numcact,numdact=self.numdact,obsres=self.obsres,actres=self.actres,pomcp_timeout=self.timeout,alpha_pomcp=alpha_a,gamma_pomcp=gamma, agent_gender=self.gender)
                
            simul_avgs = []

        print 100*"a"
        print "agent parameters: "
        self.learn_agent.print_params()
        if known_client:
            print 100*"c"
            print "client parameters: "
            self.simul_agent.print_params()


        learn_avgs=self.learn_agent.set_samples(alter)

        self.learn_agent.initialise_pomcp(alter)
        
        return (learn_avgs,simul_avgs)

    def get_recommended_next_actions(self,known_client,learn_avgs,simul_avgs):

        (learn_aab,learn_paab)=self.learn_agent.get_next_action(learn_avgs,exploreTree=True)
            
        if known_client:
            (simul_aab,simul_paab)=self.simul_agent.get_next_action(simul_avgs)
        else:
            #ignored
            simul_aab=[]
            simul_paab = 0 
        return (learn_aab,learn_paab,simul_aab,simul_paab)
            

    def simulate_alter_onestep(self,learn_aab,learn_paab,learn_observ,simul_aab,simul_paab,simul_observ,
                               known_client=False,learn_turn="client",simul_turn="agent",learn_avgs=[],simul_avgs=[]):

        #is this right? Why do I need to do this - forgot... but check on this if strange things start happening
        #removed this - I Think its useless now that everything has moved into class "Self"
        #local_ssb = copy.deepcopy(self.ssb)

        print "learner turn: ",learn_turn
        print "simulator turn: ",simul_turn

        print "agent action/client observ: ",learn_aab
        print "client action/agent observ: ",simul_aab
        #simul_observ=learn_aab
        #learn_observ=simul_aab
            
        #add environmental noise here if it is being used
        if self.env_noise>0.0:
            learn_observ=map(lambda fv: NP.random.normal(fv,self.env_noise), learn_observ)
            if known_client:
                simul_observ=map(lambda fv: NP.random.normal(fv,self.env_noise), simul_observ)
            
        print "learn observ: ",learn_observ
        print "simul observ: ",simul_observ
            
        #if using employer class, the learner gets an extra value here as the observation of x
        #so lame -- here I just use the average value ... lame lame must do this better
        if useEmployerAgent:
            learn_xobserv=[State.turnnames.index(invert_turn(learn_turn)),int(learn_avgs.x[1])]
        else:
            learn_xobserv=[State.turnnames.index(invert_turn(learn_turn))]
        simul_xobserv=[State.turnnames.index(invert_turn(simul_turn))]

        learn_avgs=self.learn_agent.propagate_forward(learn_aab,learn_observ,learn_xobserv,paab=learn_paab)
            
        if known_client:
            simul_avgs=self.simul_agent.propagate_forward(simul_aab,simul_observ,simul_xobserv)


        print "learner f is: "
        learn_avgs.print_val()
            
        if known_client:
            print "simulator f is: "
            simul_avgs.print_val()
            
        #I think these should be based on fundamentals, not transients
        (aid,cid)=self.learn_agent.get_avg_ids(learn_avgs.f)
        print "learner agent id:",aid
        print "learner client id:",cid
            
        if known_client:
            (aid,cid)=self.simul_agent.get_avg_ids(simul_avgs.f)
            print "simul agent id:",aid
            print "simul client id:",cid
                    
            
        #resample and update ssb
        for samplei in range(len(self.ssb)):
            if NP.random.random() > self.eta_value:  #replace the sample with one from fuba
                #might need a deep copy here
                newf = random.choice(self.learn_agent.samples).get_copy().f[0:3]
                self.ssb[samplei] = newf


        learn_turn = invert_turn(learn_turn)
        simul_turn = invert_turn(simul_turn)

        return (learn_avgs,simul_avgs,learn_turn,simul_turn)

    #simulates an interaction of Agent with a simulated client with identity fubc.
    #if the known_client flag is set, then do a simulation like in bayesactsim
    #between two agents with identites fuba and fubc
    #if the known_client flag is not set (default), then do a simulation of a single agent
    #who always gets a [] observation of fub
    def simulate_alter(self,T,known_client=False,learn_turn="client",simul_turn="agent"):

        (learn_avgs,simul_avgs) = self.initialise_alter(known_client)

        for t in range(T):
            print 10*"-d","iter ",t,80*"-"
            (learn_aab,learn_paab,simul_aab,simul_paab) = self.get_recommended_next_actions(known_client,learn_avgs,simul_avgs)
            (learn_avgs,simul_avgs,learn_turn,simul_turn) = self.simulate_alter_onestep(learn_aab,learn_paab,simul_aab,simul_aab,simul_paab,learn_aab,known_client,learn_turn,simul_turn,learn_avgs,simul_avgs)
            #self.updateSelfSentiment(known_client)
            #we don't update the alter models though - check this
            
        #(cnt_ags,cnt_cls)=self.learn_agent.get_all_ids()
        #print "top ten ids for agent (learner perspective):"
        #print cnt_ags[0:10]
        #print "top ten ids for client (learner perspective):"
        #print cnt_cls[0:10]
        #if known_client:
        #    (cnt_ags,cnt_cls)=self.simul_agent.get_all_ids()
        #    print "top ten ids for agent (simulator perspective):"
        #    print cnt_ags[0:10]
        #    print "top ten ids for client (simulator perspective):"
        #    print cnt_cls[0:10]
        
        self.simAlterResult =  (self.ssb,self.learn_agent.samples,self.simul_agent.samples,learn_avgs,simul_avgs,learn_turn,simul_turn)
        return self.simAlterResult

                    
    def getSimAlterResult(self):
        return self.simAlterResult
        



#-------------------------------------------------------------------------------------------------------------------
# main code
#-------------------------------------------------------------------------------------------------------------------

        
print 20*"TESTING---"

#------------------------------
#Create a new Self
me  = Self(num_samples=n_samples,eta_value=eta, etas_value=eta_s, etaf_value = eta_f, agent_gender=agent_gender, fifname=fifname)


#move this  to constructor instead
me.setPlotting(doPlotting)


#-----------------------------------------
#Create some sample initialisation engines
sampleiniter = SampleInitialiser()
if useEmployerAgent:
    esampleiniter = EmployerSampleInitialiser()
else:
    esampleiniter = SampleInitialiser()

me.useEmployerAgent = useEmployerAgent


#this is just [1.0,0.0] if we are not doing EmployerAgent
#initial turn + rest of x if needed
#since this only used to initialise the samples for fsb,ssb, 
#it doesn't matter what this is intitialised to, since those ignore x anyways
if useEmployerAgent:
    ix = [[1.0,0.0],[1.0,0.0]]
else:
    ix = [1.0,0.0]

    
#initialise the fundamental self sentiment for the Self - 
#this could be moved into the Self constructor
thesamples = esampleiniter.initialise_samples(n_fsb,fifname,agent_gender,agent_fsb,[],ix)


#read in sentiment dictionary for behaviours
fbeh_agent = {}
for gend in ["male","female"]:
    fbeh_agent[gend]=readSentiments(fbfname,gend)


fbehaviours_agent = fbeh_agent[agent_gender]

#add all the behaviours words to the list of known words
for beh in fbehaviours_agent:
    addWordToKnownWords(beh,map(lambda x: float(x), [fbehaviours_agent[beh]["e"],fbehaviours_agent[beh]["p"],fbehaviours_agent[beh]["a"]]),1.0)



#fundamental self sentiment
fsb = map(lambda x: x.f[0:3],thesamples)
#ssb will start the same as fsb, but with a small amount of added noise
ssb = map(lambda x: x.f[0:3] + NP.random.random([3])*0.1,thesamples)


#agent fundamental self sentiment
agent_ids = agent_fsb 
#set this up in the Self - possibly do all this in the constructor directly instead of above and then
#copy over to me...
me.agent_ids = agent_fsb
me.fsb = copy.deepcopy(fsb)
me.ssb = copy.deepcopy(ssb)


#to set the initial turn, set initx=[0.0,1.0] (for client) and [1.0,0.0] for agent
#here, we will start simulations for unknown identities off with client going first,
#since this is all "imaginary" anyways, and our imaginary clients know exactly who
#the agent is, so the client might as well go first to convey the most information
#if we are reading all ids from a file, it won't matter anyways
learn_turn="client"
simul_turn="agent"
#default "client"
if useEmployerAgent:
    initx0=[[0.0,1.0],[1.0,0.0]]
else:
    initx0 = [0.0,1.0]
initx0c=[1.0,0.0]
if learn_turn=="agent":
    if useEmployerAgent:
        initx0=[[1.0,0.0],[1.0,0.0]]
    else:
        initx0 = [1.0,0.0]
    initx0c=[0.0,1.0]

#------------------------------------------------------------------------
#------------------------------------------------------------------------
# Now, we do the set of initial simulations with the known
# alters only and the stranger to get the estimates of \ssb^{\dagger}[j]
#------------------------------------------------------------------------
#------------------------------------------------------------------------


#---------------------------------------------------------
# read from a file if the initial phase has been done previously
# Since these are each stored in a file with the client's name,
# we check to see if that file exists, and just read from it if it is there,
# otherwise, we do the simulation and record the result in a new file

# first, add a "stranger" id
if useStranger:
    client_names     += ["stranger"]
    client_ids       += [None]
    client_genders   += ["female"]  # should really be 50/50 male female, but we are operating in a strictly all-female world for now

    client_agent_ids += [None]
    client_variances += [None]

#now, loop over all known clients and the stranger
for alter_index in range(len(client_names)):
    alter = client_names[alter_index]
    fname = identityFileBaseName+alter+".txt"
    me.theAlterIds[alter] = client_ids[alter_index]
    try:
        fp = open(fname, 'r')
        (altername,numsamples) = fp.readline().split()
        if not altername == alter+"ssb":
            print "error in file"
        print "numsamples:  ",numsamples
        me.ssbClients[alter] = []
        for i in range(int(numsamples)):
            f = map(lambda x: float(x), fp.readline().split())
            me.ssbClients[alter].append(f)


        (altername,numsamples) = fp.readline().split()
        if not altername == alter+"agentf":
            print "error in file"
        print "numsamples:  ",numsamples
        me.theAlters[alter] = []
        for i in range(int(numsamples)):
            f = map(lambda x: float(x), fp.readline().split())
            x = map(lambda x: int(x), fp.readline().split())    # not necessarily always an int ~~~!!!!!
            me.theAlters[alter].append(State(NP.asarray(f),NP.asarray(f),x,1.0))


        (altername,numsamples) = fp.readline().split()
        if not altername == alter+"clientf":
            print "error in file"
        print "numsamples:  ",numsamples
        me.theClientAlters[alter] = []
        for i in range(int(numsamples)):
            f = map(lambda x: float(x), fp.readline().split())
            x = map(lambda x: int(x), fp.readline().split())    # not necessarily always an int ~~~!!!!!
            me.theClientAlters[alter].append(State(NP.asarray(f),NP.asarray(f),x,1.0))

        #lat line has the gender
        fg = fp.readline().split()
        if fg[1] == "None":
            me.theClientGenders[alter] = []
        else:
            me.theClientGenders[alter] = fg[1]
        fp.close()
    except IOError:
        print "File ",fname," does not exist, simulating for ",alter
        T = identitySimulationT
        #The file does not exist, so we are going to simulate each alter to find the situational sentiment
        #expected to be caused by that alter.    
        #here, we simply intitialise f_a[] and f_c[] arrays (as me.theAlters and me.theClientAlters), 
        #and later on we will do the interactions
        cid = client_ids[alter_index]  #None for stranger
        cgen = client_genders[alter_index]
        caid = client_agent_ids[alter_index]  #None for stranger
        cvar = client_variances[alter_index]  #None for stranger
        cavar = client_agent_variances[alter_index]  #None for stranger
        #f_a[j], f_c[j]
        if not alter == "stranger":
            me.theAlters[alter] = esampleiniter.initialise_samples(me.num_samples,fifname,agent_gender,agent_ids,[[1.0,cid,cvar]],initx=initx0)
        
            #f_a and f_c for client simulations
            me.theClientAlters[alter] = sampleiniter.initialise_samples(me.num_samples,fifname,cgen,[[1.0,cid,cvar]],[[1.0,caid,cavar]],initx=initx0c)   #same variance for all
            me.theClientGenders[alter] = cgen

        else:
            #one stranger
            me.theAlters[alter] = esampleiniter.initialise_samples(me.num_samples,fifname,agent_gender,agent_ids,[],initx=initx0)
            me.theClientAlters[alter] = []
            me.theClientGenders[alter] = []

        #writeMatlab('jerrysusan.m','0',fsb,[],me.theAlters,me.theClientAlters,wmode='w')

        if alter=="stranger":
            known_client=False
        else:
            known_client=True

        print 100*"="
        print "the alter: ",alter
        
        #always this way to start for simulations
        learn_turn="client"
        simul_turn = "agent"
        
        me.bestAlter = alter
        #----- plotting stuff inserted here --- don't do this during simulation as it does not work
        if False and doPlotting:
            (learn_avgs,simul_avgs) = me.initialise_alter(True)

            app = wx.App(redirect=False)
            frame = SimulFrame(None,"Bayesactself",T,known_client=known_client,run_once=True,learn_avgs=learn_avgs,simul_avgs=simul_avgs,learn_turn=learn_turn,simul_turn=simul_turn)
            
            #need to set this to one temporarily because the ssbClient[alter] will not exist in findBestAlter
            etas_value_save = me.etas_value
            me.etas_value = 1.0
            frame.setMe(me)
            me.etas_value = etas_value_save
            frame.Show()
            app.SetTopWindow(frame)
            print 30*"chick"
            ## PLOTTING DOES NOT WORK for learning sentiments because I can't get this app to close
            app.MainLoop()
            print 30*"chicken"
            app.Destroy()

            print 30*"rooster"

            ### CHECK TO MAKE SURE me.theAlters and me.theClientAlters ARE CHANGING
            #record and store the final sentiments - this is f_a[] and f_c[]
            (me.ssbClients[alter],me.theAlters[alter],me.theClientAlters[alter],learn_avgs,simul_avgs,learn_turn,simul_turn) = me.getSimAlterResult()
        else:
            #------ used to be just this: 
            (me.ssbClients[alter],me.theAlters[alter],me.theClientAlters[alter],learn_avgs,simul_avgs,learn_turn,simul_turn) = me.simulate_alter(T,known_client,learn_turn,simul_turn)

        #write out final fs and situational self-sentiments to a file we can read back in with Python
        #this should really be to store to a database
        
        fp = open(fname, 'w')
        fp.write(alter+"ssb"+" "+str(len(me.ssbClients[alter]))+"\n")
        for x in me.ssbClients[alter]:
            for y in x:
                fp.write(str(y)+" ")
            fp.write("\n")

        fp.write(alter+"agentf"+" "+str(len(me.theAlters[alter]))+"\n")
        for x in me.theAlters[alter]:
            for y in x.f:
                fp.write(str(y)+" ")
            fp.write("\n")
            for y in x.x:
                fp.write(str(y)+" ")
            fp.write("\n")
            
        fp.write(alter+"clientf"+" "+str(len(me.theClientAlters[alter]))+"\n")
        for x in me.theClientAlters[alter]:
            for y in x.f:
                fp.write(str(y)+" ")
            fp.write("\n")
            for y in x.x:
                fp.write(str(y)+" ")
            fp.write("\n")
            
    
        fp.write(alter+"clientg"+" "+me.theClientGenders[alter]+"\n")
        fp.close()




        
#write out final fs and situational self-sentiments for matlab plotting only
#writeMatlab('jerrysusan.m','4',[],me.ssbClients,me.theAlters,me.theClientAlters,wmode='a')



## some generic old stuff to test out the kldivergence stuff
'''--------------------------------------------------------------------------------------------------------------------
etas = 0.5
kldiv={}

for alter in me.theAlters:
    print 50*"^"
    print "the alter: ",alter
    print 50*"-"
    
    #now, we combine the situational of two identities and see which one works bettter
    kldiv[alter]={}
    for alter1 in me.theAlters:
        #combine me.ssbClients[alter] with me.ssbClients[alter1] and see which one works best
        ssb_mixed = copy.deepcopy(me.ssbClients[alter]) 
        #add a bit of random noise so the KL divergence doesn't diverge
        for i in range(len(ssb_mixed)):
            if NP.random.random() > etas:
                #replace with sample from me.ssbClients[alter1]
                ssb_mixed[i] = copy.deepcopy(me.ssbClients[alter1][i])
        ssb_mixed = map(lambda x: x + NP.random.random([3])*0.01,ssb_mixed)
        #this should be compared to fsb I think, not me.ssbClients[alter1] ? 
        # was like this: kldiv[alter][alter1] = kldivergence(ssb_mixed,me.ssbClients[alter1])
        #should be 
        kldiv[alter][alter1] = kldivergence(ssb_mixed,fsb)
        print "after interacting with ",alter," , interaction with ", alter1," will have a divergence of: ",kldiv[alter][alter1]
        

f = open('jerrysusan.m', 'a')
f.write("kldivergences=[\n")
for alter in me.theAlters:
    for alter1 in me.theAlters:
        f.write(str(kldiv[alter][alter1])+" ")
    f.write(";\n")
f.write("];\n")

f.close()
--------------------------------------------------------------------------------------------------------------------'''



#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
# START OF MAIN LOOP
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

 
rseed = NP.random.randint(0,382948932)
rseed = 73666388

print "random seeed is : ",rseed
NP.random.seed(rseed)

#now start the main loop
#number of steps to take during each interaction - must be a multiple of two to always restore agent turn as first
#however, the turn-taking issue has not really been properly dealt with, since a client could take multiple turns, etc
# is really a property of the SimulFrame I think - its something to do with the nuts and bolts of a simulation
T=2


#initialise all the alters and find first actions
me.initialiseAllAlters()

#----- plotting stuff inserted here
if doPlotting:
    app = wx.App(redirect=False)
    frame = SimulFrame(None,"Bayesactself",T)
    frame.setMe(me)
    frame.Show()
    app.MainLoop()
    app.Destroy()
        

