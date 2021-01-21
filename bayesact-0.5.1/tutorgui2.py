"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Tutor GUI example
Author: Jesse Hoey
September 2013
Use for research purposes only.
Please do not re-distribute.
Any commerical uses strictly forbidden.
Research sponsored by the Natural Sciences and Engineering Council of Canada (NSERC).
use python2.6
I started this GUI from test7.py by Robin Dunn 
may have to start this as:  arch -i386 python2.6 tutorgui2.py 
see README for details
------------------------------------------------------------------------------------------"""


from discretetutor import *
#import the wxPython GUI package
import wx
import csv



#file with the fundamentals for behaviors for the tutor
fbfname_tutor="fbehaviours-tutor.dat"

fbfname_student="fbehaviours-student.dat"

falabels_file="action-sayings.dat"

agent_gender="male"
client_gender="male"


#file with the fundamentals for identities
fifname="fidentities.dat"


#use pomcp for planning (default use heuristic/greedy method)
#doesn't work yet - must deal with get_prop_action and get_next_action in discretetutor.py
use_pomcp=False

#parameters for pomcp
numcact=5  #user defined 
numdact=3  #3 skill levels
obsres=1.0
actres=1.0
timeout=10.0


#read in action sayings
fb=file(falabels_file,"rU")
cread = csv.reader(fb)
action_sayings={}
student_as={}
tutor_as={}
for x in cread:
    action_sayings[x[0]]=x[1]
fb.close()


student_as = dict((k,v) for k, v in action_sayings.iteritems() if k.startswith("s"))
tutor_as = dict((k,v) for k, v in action_sayings.iteritems() if k.startswith("t"))

student_as_q={}
student_as_q[True]=dict((k,v) for k, v in student_as.iteritems() if k.startswith("sa"))
student_as_q[False]=dict((k,v) for k, v in student_as.iteritems() if k.startswith("sb"))
#read in client behaviours file
addto_client=0
if client_gender=="female":
    addto_client=3
fbehaviours_client={}
fb=file(fbfname_student,"rU")

for x in set(fb):
    y=[re.sub(r"\s+","_",i.strip()) for i in x.split(",")]
    fbehaviours_client[y[0]]={"e":y[1+addto_client],"p":y[2+addto_client],"a":y[3+addto_client]}
fb.close()

fbehaviours_client_q={}
fbehaviours_client_q[True]=dict((k,v) for k, v in fbehaviours_client.iteritems() if k.startswith("a"))
fbehaviours_client_q[False]=dict((k,v) for k, v in fbehaviours_client.iteritems() if k.startswith("b"))

#open the input file for fundamentals of behaviours
##this should really be done by the application, not here
addto_agent=0
if agent_gender=="female":
    addto_agent=3
fbehaviours_agent={}

fb=file(fbfname_tutor,"rU")
for x in set(fb):
    y=[re.sub(r"\s+","_",i.strip()) for i in x.split(",")]
    #1,2,3 areEPA for males, 4,5,6 are EPA for females
    #for males
    fbehaviours_agent[y[0]]={"e":y[1+addto_agent],
                             "p":y[2+addto_agent],
                             "a":y[3+addto_agent]}
fb.close()

fbehaviours_agent_q={}
fbehaviours_agent_q[True]=dict((k,v) for k, v in fbehaviours_agent.iteritems() if k.startswith("a"))
fbehaviours_agent_q[False]=dict((k,v) for k, v in fbehaviours_agent.iteritems() if k.startswith("b"))


def print_acts(sas):
    for s in sas.keys():
        print s[1:]," : ",sas[s]

#a function to ask the client for an action to take and map it to an EPA value
def ask_client(fbeh,sas):
    print "pick an action and type in its label: "
    print_acts(sas)
    while True:
        cact = raw_input("enter client action: ")
        #replace all spaces with _
        cact=re.sub(r"\s+","_",cact.strip())
        if cact in fbeh.keys():
            break
        else:
            print "not found, try again"
    observ=map(lambda x: float (x), [fbeh[cact]["e"],fbeh[cact]["p"],fbeh[cact]["a"]])
    print "client sez: ",cact
    return observ

#basic stub for a function that asks a question and gets an answer, checks the answer
#returns the question (as a string)
#and a list of answers (otheransw)
#and an index into that list with the correct answer
#OR, a question (a string)
#and an answer (a string)
#last return value in this case is []
#THIS SECOND OPTION is not implemented yet!
def ask_question(difficulty_level):

    #normally pick based on difficulty_level here (0,1,2) for (easy, medium, hard)
    if difficulty_level<1:
        maxadd=10
    elif difficulty_level<1.5:
        maxadd=20
    else:
        maxadd=100
    x=NP.random.randint(maxadd)
    y=NP.random.randint(maxadd)
    quest=str(x)+" + "+str(y)+"= ? "
    theintansw=x+y
    theansw=str(theintansw)
    otheransw=map(lambda x: str(x),filter(lambda x: not x==theintansw,range(maxadd*2)))
    NP.random.shuffle(otheransw)
    #pick first 5
    otheransw=otheransw[0:5]
    #add the correct answer back in 
    otheransw.append(theansw)
    #remove duplicates - not needed anymore
    #otheransw = reduce(lambda x, y: x if y in x else x + [y], otheransw, [])
    NP.random.shuffle(otheransw)
    answ=otheransw.index(theansw)
    return (quest,answ,otheransw)

textanswer=False

class ClientSayPanel(wx.Panel):
    def __init__(self, parent,correct):
        wx.Panel.__init__(self, parent=parent)
 
 
        self.sayButtons=[]
        #student_as is global
        self.buttonKeys=[]
        keyindex=0
        for sas in student_as_q[correct]:
            self.buttonKeys.append(sas)
            self.sayButtons.append(wx.Button(self, keyindex, student_as_q[correct][sas]))
            self.Bind(wx.EVT_BUTTON, parent.OnSay, self.sayButtons[-1])
            keyindex+=1

        clientSaySizer=wx.FlexGridSizer(3,3,hgap=5,vgap=5)
        for butt in self.sayButtons:
            clientSaySizer.Add(butt)
        self.SetSizer(clientSaySizer)
 
# Create a new frame class, derived from the wxPython Frame.
class MyFrame(wx.Frame):

    def __init__(self, parent, id, title,maxit=10):
        # First, call the base class' __init__ method to create the frame
        wx.Frame.__init__(self, parent, id, title)

        self.maxIterations=maxit


        menuBar = wx.MenuBar()
        file = wx.Menu()
        m_exit = file.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Close window and exit program.")
        self.Bind(wx.EVT_MENU, self.OnClose, m_exit)
        menuBar.Append(file, "&File")
        about = wx.Menu()
        m_about = about.Append(wx.ID_ABOUT, "&About", "Information about this program")
        self.Bind(wx.EVT_MENU, self.OnAbout, m_about)
        menuBar.Append(about, "&Help")
        self.SetMenuBar(menuBar)
        self.statusbar = self.CreateStatusBar()

        # Add a panel and 
        panel = wx.Panel(self, -1)

        #the first question
        (self.thequestion,self.theanswer,self.listofanswers)=ask_question(difficulty_level)

        # text box for question
        self.questionLabel = wx.StaticText(panel, -1, self.thequestion)
        #text box for agent saying
        self.agentSayLabel = wx.StaticText(panel, -1, "\""+agentSay+"\"")
        self.agentSayTitleLabel=wx.StaticText(panel, -1, "Tutor Says: ")
        #text box for messgaes
        self.messageLabel = wx.StaticText(panel, -1, "")


        font = wx.Font(24, wx.DECORATIVE, wx.ITALIC, wx.BOLD)
        font2 = wx.Font(24, wx.ROMAN, wx.NORMAL, wx.BOLD)
        self.agentSayLabel.SetFont(font)
        self.agentSayTitleLabel.SetFont(font2)
        self.questionLabel.SetFont(font2)
        # another textbox for the answer if there are no options given
        # not implemented yet
        if self.listofanswers==[]:
            self.answerField = wx.TextCtrl(panel,-1)
            self.Bind(wx.EVT_TEXT, self.OnChange, self.answerField)
            self.Bind(wx.EVT_CHAR, self.OnKeyPress, self.answerField)
        else:
            self.answers=[]
            firstButton=True
            for answer in self.listofanswers:
                if firstButton:
                    self.answers.append(wx.RadioButton(panel, label=answer+"   ", style = wx.RB_GROUP))
                    firstButton=False
                else:
                    self.answers.append(wx.RadioButton(panel, label=answer+"   "))

        # a submit button
        self.submitButton = wx.Button(panel,-1,"check my answer")
        self.Bind(wx.EVT_BUTTON, self.OnSubmit, self.submitButton)

        self.panel = panel

        self.clSayPanel={}
        for correct in (True,False):
            self.clSayPanel[correct]=ClientSayPanel(self,correct)
   
        # Use some sizers for layout of the widgets
        agentSaySizer=wx.BoxSizer(wx.HORIZONTAL)
        agentSaySizer.Add(self.agentSayTitleLabel)
        agentSaySizer.Add(self.agentSayLabel)

        qaSizer=wx.BoxSizer(wx.VERTICAL)
        questionSizer=wx.BoxSizer(wx.VERTICAL)
        questionSizer.Add(self.questionLabel)
        answersSizer = wx.BoxSizer(wx.VERTICAL)
        answerSizer=wx.FlexGridSizer(3,4,hgap=5, vgap=5)
        if textanswer:
            answerSizer.Add(self.answerField)
        else:
            for answer in self.answers:
                answerSizer.Add(answer)


        answersSizer.Add(answerSizer)
        answersSizer.Add(self.submitButton)
        qaSizer.Add(questionSizer)
        qaSizer.Add(answersSizer)
        self.qaSaySizer=wx.BoxSizer(wx.HORIZONTAL)
        self.qaSaySizer.Add(qaSizer)

        for correct in (True,False):
            self.qaSaySizer.Add(self.clSayPanel[correct],1,wx.ALL,25)
        self.clSayPanel[True].Hide()
        border = wx.BoxSizer(wx.VERTICAL)
        border.Add(agentSaySizer, 0, wx.ALL, 15)
        border.Add(self.qaSaySizer, 1, wx.ALL, 15)
        panel.SetSizerAndFit(border)
        self.SetSizer(border)
        #first fit the window
        self.Fit()
        #then hide the saying buttons
        for correct in (True,False):
            self.clSayPanel[correct].Hide()
        self.updateAgent=False
        self.iteration=0
        wx.EVT_IDLE(self, self.OnIdle) 

    def OnSubmit(self,event):
        #get the answer they selected
        whichanswer=map(lambda x: x.GetValue(),self.answers)
        if not (True in whichanswer):
            self.agentSayLabel.SetLabel("\"select an answer first\"")
            return

        student_answer=whichanswer.index(True)
        self.correct=(student_answer==self.theanswer)

        if self.correct:
            self.agentSayLabel.SetLabel("\"correct!\"")
        else:
            self.agentSayLabel.SetLabel("\"incorrect, correct answer was: "+str(self.listofanswers[self.theanswer]+"\""))

            #self.agentSayLabel.SetLabel("Now select an option to continue...")
        #show the buttons
        print "client was correct?: ",self.correct
        self.clSayPanel[self.correct].Show()
        self.submitButton.Hide()
        self.Layout()

    def OnClose(self, event):
        self.Destroy()
    def OnDone(self):
        dlg = wx.MessageDialog(self,"Done! Thanks for trying the Affective Tutor", "", wx.OK|wx.CANCEL|wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            self.Close()

    def OnAbout(self, event):
        dlg = wx.MessageDialog(self,"Bayesian Affective Tutor, version 1.0","",wx.OK|wx.CANCEL|wx.ICON_QUESTION)
        dlg.ShowModal()
        dlg.Destroy()
         
    def OnSay(self,event):
        self.agentSayLabel.SetLabel("\"Thinking....\"")
        #self.correctLabel.SetLabel("")
        
        #which button did they push? - use self.correct here to get the right set
        cact=self.clSayPanel[self.correct].buttonKeys[event.GetId()]
        print "client said: ",cact
        cact=cact[1:]
        self.clientObserv=map(lambda x: float (x), [fbehaviours_client[cact]["e"],fbehaviours_client[cact]["p"],fbehaviours_client[cact]["a"]])
        self.updateAgent=True

        self.iteration += 1
        if self.iteration > self.maxIterations:
            self.OnDone()

    def OnIdle(self, event):
        if not self.updateAgent:
            return
        self.updateAgent=False
        #get the agent action - also include whether they got it right or wrong here (self.correct)
        print "updating agent with xobserv: ",[State.turnnames.index("client"),int(self.correct)]
        
        #(affective_action,propositional_action)=agent.update(self.clientObserv,

        #update on client turn
        #these zeros are arbitrary and only there to match with the ones generated by the oracle 
        #this arbitrariness should really be fixed ....
        avgs=agent.propagate_forward([0.0,0.0,0.0],self.clientObserv,[State.turnnames.index("agent"),int(self.correct)],0)
        print "agent average sentiments: "
        avgs.print_val()

        #then on agent turn
        #propositional_action = agent.get_prop_action()
        (affective_action,propositional_action)=agent.get_next_action(avgs,exploreTree=True)   

        #propagate for agent
        print "agent action: ",affect_action, " propositional action: ",propositional_action

        avgs=agent.propagate_forward(affect_action,[],[State.turnnames.index("client"),0],propositional_action)
        print "agent average sentiments: "
        avgs.print_val()


        difficulty_level=propositional_action
        print "difficulty level is : ",difficulty_level
        #here we would implement the random policy
        aab = findNearestBehaviour(affective_action,fbehaviours_agent_q[self.correct])
        agentSay=tutor_as["t"+aab]
        self.agentSayLabel.SetLabel("\""+agentSay+"\"")
        
        #replace the question with a new one
        (self.thequestion,self.theanswer,self.listofanswers)=ask_question(difficulty_level)

        i=0
        for answer in self.listofanswers:
            self.answers[i].SetLabel(answer)
            self.answers[i].SetValue(False)
            i=i+1

        for correct in (True,False):
            self.clSayPanel[correct].Hide()
        self.questionLabel.SetLabel(self.thequestion)
        self.submitButton.Show()
        self.Layout()



# Every wxWidgets application must have a class derived from wx.App
class MyApp(wx.App):
    maxIterations=10
    def setMaxIt(self,maxit):
        self.maxIterations=maxit
    # wxWindows calls this method to initialize the application
    def OnInit(self):

        # Create an instance of our customized Frame class
        frame = MyFrame(None, -1, title="Bayesact Tutor",maxit=self.maxIterations)
        frame.Show(True)

        # Tell wxWindows that this is our main window
        self.SetTopWindow(frame)

        # Return a success flag
        return True

#-----------------------------------------------------------------------------------------------------------------------------
#simulation start
#-----------------------------------------------------------------------------------------------------------------------------

#for repeatability
rseed = NP.random.randint(0,382948932)
#rseed = 101954541
#should print to a log
print "random seeed is : ",rseed

#----------------- experiment set-up -----------------------------------------------------------------
tau_init=[]
#----------------- experiment type 1 -----------------------------------------------------------------
#this experiment, the tutor starts with a single id as exactly the client
#this reproduces interact simulations closely.
#NOTE: it may not be *exactly* the same as the interact simulations because I am using a set of
#samples to represent each belief state.  It is possible to crank the variances down on the simulation
#so it more closely follows interact, but this more easily becomes unstable and breaks the simulation.
#using the default values you get a good simulation which is quite close to interact
#student and tutor
tau_init.append([1.51, 1.45, -0.18, 0.00, 0.0, 0.0, 1.49, 0.31, 0.75])
#champion and bore - another one you can try
#tau_init.append([2.47, 2.52, 2.23, 0.0, 0.0, 0.0,  -1.26, -0.95, -1.63])
num_samples=500
initial_id_variance_client=0.1
prop_init=[1.0]

#------------------ experiment type 2 ----------------------------------------------------------------
#this experiment, the tutor starts with three possible identities for the client, equally weighted
#university student, graduate student and student
#university student, 1.01, 0.34, 0.94, 2.28, 0.53, 2.43, 11 000010000 000
#graduate student, 1.4, 0.94, 0.26, 1.49, 1.46, 0.7, 11 000010000 000
#student, 1.49, 0.31, 0.75, 2.22, 1.12, 1.5, 11 000010000 000
#tau_init.append([1.51, 1.45, -0.18, 0.00, 0.0, 0.0, 1.49, 0.31, 0.75])
#tau_init.append([1.51, 1.45, -0.18, 0.00, 0.0, 0.0, 1.4, 0.94, 0.26])
#tau_init.append([1.51, 1.45, -0.18, 0.00, 0.0, 0.0, 1.01, 0.34, 0.94])
#num_samples=1000
#initial_id_variance_client=0.2
#prop_init=[0.333,0.334,0.333]

#------------------ experiment type 3----------------------------------------------------------------
# assume an identity and see if the tutor can figure it out
# may be slow due to the number of samples needed
#this experiment uses a tutor that starts off with a default (0,0,0) identity
#for the client and learns what the true identity is
#tau_init.append([1.51, 1.45, -0.18, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0])
#num_samples=2000
#initial_id_variance_client=0.5
#prop_init=[1.0]

#------------------- end setup ---------------------------------------------------------------

NP.random.seed(rseed)

#this is a "tutoring agent" only because of the "x" dynamics, which are ignored so far since I am not computing rewards yet
agent=DiscreteTutoringAgent(N=num_samples,beta_value_client_init=initial_id_variance_client,identities_file=fifname,behaviours_file=fbfname_tutor,use_pomcp=use_pomcp,numcact=numcact,numdact=numdact,obsres=obsres,actres=actres,pomcp_timeout=timeout)

initial_turn="agent"

#this is the first action to take - initx=[] here because that will mean the agent will draw a sample from the initial distribution
avgs=agent.initialise_array(tau_init,prop_init,[])

print "getting agent action..."
(affect_action,prop_action)=agent.get_next_action(avgs)   

#propagate for agent
print "agent action: ",affect_action, " prop action: ",prop_action

avgs=agent.propagate_forward(affect_action,[],[State.turnnames.index("client"),0],prop_action)
print "agent average sentiments: "
avgs.print_val()



aab = findNearestBehaviour(affect_action,fbehaviours_agent)
agentSay=tutor_as["t"+aab]
difficulty_level=prop_action

#(1,"outputlog.txt") argument says to print stdout on the sccreen - if 0 it does not do this
app = MyApp(0)     # Create an instance of the application class
app.setMaxIt(10)
app.MainLoop()     # Tell it to start processing events


