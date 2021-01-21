"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Interactive Example
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
from bayesactemot import *
import getopt
import sys
import threading
sys.path.append("./gui/")
from cEnum import eTurn


class cBayesactInteractive(object):
    def __init__(self, argv, plotter=None):
        #NP.set_printoptions(precision=5)
        #NP.set_printoptions(suppress=True)
        NP.set_printoptions(linewidth=10000)



        #-----------------------------------------------------------------------------------------------------------------------------
        #user-defined parameters
        #-----------------------------------------------------------------------------------------------------------------------------
        #agent knowledge of client id:
        #0 : nothing
        #1 : one of a selection of  num_confusers+1 randoms
        #2 : exactly - use this to mimic interact
        #3 : same as 0 but also agent does not know its own id
        self.agent_knowledge=0

        # agent gender
        self.agent_gender="male"

        # client gender
        self.client_gender="male"

        #possibly set the agent id to be something
        self.agent_id="tutor"
        #if not in database (including "") then it is a randomly drawn id
        #agent_id=""

        #can also set the client id here if agent_knowledge = 2 (knows id of client - see above)
        #if agent_knowledge is 0 then this is ignored
        #client_id = "professor"
        self.client_id = "student"

        #who goes first?
        self.initial_turn="agent"

        #how often do we want to see the full id sets learned by the agent
        self.get_full_id_rate=10


        #do we want to try to mimic interact?
        self.mimic_interact=False

        #use pomcp for planning (default use heuristic/greedy method)
        self.use_pomcp=False

        #parameters for pomcp
        #number of continuous actions we wish to sample -
        #this is user-defined and is an important parameter
        #larger numbers mean bigger, slower, more accurate,  planning trees
        self.numcact=5
        #number of discrete (propositional) actions
        #this should be set according to the domain, and is 1 for this generic class
        #one discrete action really means no choice
        self.numdact=1
        #observation resolution when building pomcp plan tree
        self.obsres=1.0
        #action resolution when buildling pomcp plan tree
        self.actres=0.1
        #timeout used for POMCP
        self.timeout=5.0

        #-----------------------------------------------------------------------------------------------------------------------------
        #these parameters can be tuned, but using caution
        #-----------------------------------------------------------------------------------------------------------------------------
        #agent's ability to change identities - higher means it will shape-shift more
        self.bvagent=0.0001
        #agent's belief about the client's ability to change identities - higher means it will shape-shift more
        self.bvclient=0.0001


        
        #-----------------------------------------------------------------------------------------------------------------------------
        #these parameters can be tuned, but will generally work "out of the box" for a basic simulation
        #-----------------------------------------------------------------------------------------------------------------------------

        #behaviours file
        self.fbfname="fbehaviours.dat"

        #identities file
        self.fifname="fidentities.dat"

        #get some key parameters from the command line
        #set much larger to mimic interact (>5000)
        self.num_samples=500


        # use 0.0 for mimic interact simulations
        #roughening_noise=0.0
        self.roughening_noise=self.num_samples**(-1.0/3.0)


        #the observation noise
        #set to 0.05 or less to mimic interact
        self.obs_noise=0.1

        if self.mimic_interact:
            self.mimicInteract()

        self.gamma_value=self.obs_noise


        #for emotional agent - gamma value for observations of emotion
        self.gammae_value = 0.1


        #do we print out all the samples each time
        self.learn_verbose=False

        #for repeatability
        self.rseed = NP.random.randint(0,382948932)
        #rseed=271887164

        NP.random.seed(self.rseed)

        #if true, will use an EmotionalAgent from bayesactemot.py
        #rather than an Agent from bayesact.py
        self.useEmotionalAgent = False
        
        self.helpstring="Bayesact interactive simulator (1 agent, 1 human) usage:\n bayesactinteractive.py\n\t -n <number of samples (default 500)>\n\t -a <agent knowledge (0,1,2) (Default 2)>\n\t -r <roughening noise: default n^(-1/3) - to use no roughening ('full' method), specify 0>\n\t -g <gamma_value (default 0.1)>\n\t -i <agent id label: default randomly chosen>\n\t -j <client id label: default randomly chosen>\n\t -k <agent gender (default: male) - only works if agent_id is specified with -i>\n\t -l (client gender (default: male) only works if client_id is specified with -j>\n\t -m <gammae_value (default 0.0 or emotions not used)>\n\t"

        try:
            self.opts, self.args = getopt.getopt(argv[1:],"huvon:t:x:a:c:d:r:e:g:m:i:j:k:l:",["help","n=","t=","x=","c=","a=","u=","d=","r=","e=","g=","m=","i=","j=","k=","l="])
        except getopt.GetoptError:
            print self.helpstring
            sys.exit(2)
        for opt, arg in self.opts:
            if opt == '-h':
                print self.helpstring
                sys.exit()
            elif opt == "-v":
                self.learn_verbose=True
            elif opt in ("-n", "--numsamples"):
                self.num_samples = int(arg)
            elif opt in ("-a", "--agentknowledge"):
                self.agent_knowledge = int(arg)
            elif opt in ("-r", "--roughen"):
                self.roughening_noise=float(arg)
            elif opt in ("-g", "--gamma"):
                self.gamma_value=float(arg)
            elif opt in ("-m", "--gammae"):
                self.gammae_value=float(arg)
                self.useEmotionalAgent = True
            elif opt in ("-i", "--agentid"):
                self.agent_id=arg
            elif opt in ("-j", "--clientid"):
                self.client_id=arg
            elif opt in ("-k", "--agentgender"):
                self.agent_gender=arg
            elif opt in ("-l", "--clientgender"):
                self.client_gender=arg


        self.plotter=plotter



    def mimicInteract(self):
        self.bvagent_init=0.000001
        self.bvclient_init=0.000001
        self.bvagent=0.00001
        self.bvclient=0.00001
        self.agent_knowledge=2
        self.num_samples=10000
        self.roughening_noise=0.0
        self.obs_noise=0.01

        #FIXME: what does this do?
        self.num_action_samples=10000



    def startBayesactInteractive(self):
        print "random seeed is : ",self.rseed

        #-----------------------------------------------------------------------------------------------------------------------------
        #code start - here there be dragons - only hackers should proceed, with caution
        #-----------------------------------------------------------------------------------------------------------------------------

        def is_array_of_floats(possarray):
            try:
                parr = [float(x) for x in possarray.split(',')]
            except ValueError:
                return False
            if len(parr)==3:
                return parr
            return False



        #a function to ask the client for a value (either string label or EPA vector)
        # to take and map it to an EPA value. string labels and EPA are related through dictionary fbeh
        # inputs who and what are just for printing messages describing what is going on
        # inputs sugg_act and epaact are a default label or vector the use can choose one of easily
        def ask_client(fbeh,sugg_act='',epaact=[],who="agent",what="action"):
            sst=what+" to enact/display/take for "+who+" ('?' shows options): "

            sstnc="You can now either :\n"
            sstnc+="- pick an "+what+" and type in its label\n"
            sstnc+="- type 'quit' to stop the simulation\n"
            sstnc+="- type a comma separated list of three numbers to specify the "+what+" as E,P,A values\n"
            if not sugg_act=='':
                sstnc+="- hit 'enter' to take default "+what+" with label: "+sugg_act+"\n"
            if not epaact=='':
                sstnc+="- type any digit (0-9) to take suggsted "+what+": "+str(epaact)


            while True:
                cact = raw_input(sst)
                if cact=='quit':
                    return []
                elif cact=='' and not sugg_act=='':
                    cact=sugg_act
                    break
                elif cact.isdigit() and not epaact==[]:
                    return epaact
                elif re.sub(r"\s+","_",cact.strip()) in fbeh.keys():
                    cact=re.sub(r"\s+","_",cact.strip())
                    break
                elif cact=="?":
                    print sstnc
                elif is_array_of_floats(cact):
                    return [float(x) for x in cact.split(',')]
                else:
                    print "incorrect or not found, try again. '?' shows options"
            observ=map(lambda x: float (x), [fbeh[cact]["e"],fbeh[cact]["p"],fbeh[cact]["a"]])
            print "client said: ",observ
            return observ

        fbehaviours_agent=readSentiments(self.fbfname,self.agent_gender)
        fbehaviours_client=readSentiments(self.fbfname,self.client_gender)

        (agent_mean_ids,agent_cov_ids)=getIdentityStats(self.fifname,self.agent_gender)
        (client_mean_ids,client_cov_ids)=getIdentityStats(self.fifname,self.client_gender)

        #the mean and covariance of IDs for male agents as taken from the databases
        #should do this automatically in python based on actual genders of client/agent....as above now
        #mean_ids=NP.array([0.40760,0.40548,0.45564])
        #$cov_ids=NP.array([[2.10735,1.01121, 0.48442],[1.01121,1.22836,0.55593],[0.48442,0.55593,0.77040]])

        #the actual (true) ids drawn from the distribution over ids, if not set to something in particular
        self.agent_id=getIdentity(self.fifname,self.agent_id,self.agent_gender)
        if self.agent_id==[]:
            self.agent_id=NP.random.multivariate_normal(agent_mean_ids,agent_cov_ids)
        self.agent_id=NP.asarray([self.agent_id]).transpose()

        #here we get the identity of the client *as seen by the agent*
        self.client_id=getIdentity(self.fifname,self.client_id,self.agent_gender)
        if self.client_id==[]:
            self.client_id =  NP.random.multivariate_normal(client_mean_ids,client_cov_ids)
        self.client_id=NP.asarray([self.client_id]).transpose()

        #get initial sets of parameters for agent
        (learn_tau_init,learn_prop_init,learn_beta_client_init,learn_beta_agent_init)=init_id(self.agent_knowledge,self.agent_id,self.client_id,client_mean_ids)

        #overwrite these values for the interactive script only
        #do this for mimicking interact
        if self.mimic_interact:
            learn_beta_agent_init=self.bvagent_init
            learn_beta_client_init=self.bvclient_init


        #initial x - only contains the turn for a default agent (no other x components)
        learn_initx=self.initial_turn

        #get the agent - can use some other subclass here if wanted
        if self.useEmotionalAgent:
            learn_agent=EmotionalAgent(N=self.num_samples,alpha_value=1.0,
                                       gammae_value=self.gammae_value, gamma_value=self.gamma_value,
                                       beta_value_agent=self.bvagent,beta_value_client=self.bvclient,
                                       beta_value_client_init=learn_beta_client_init,beta_value_agent_init=learn_beta_agent_init,
                                       client_gender=self.client_gender,agent_gender=self.agent_gender,
                                       agent_rough=self.roughening_noise,client_rough=self.roughening_noise,use_pomcp=self.use_pomcp,
                                       init_turn=self.initial_turn,numcact=self.numcact,numdact=self.numdact,obsres=self.obsres,
                                       actres=self.actres,pomcp_timeout=self.timeout)
        else:
            learn_agent=Agent(N=self.num_samples,alpha_value=1.0,
                              gamma_value=self.gamma_value,beta_value_agent=self.bvagent,beta_value_client=self.bvclient,
                              beta_value_client_init=learn_beta_client_init,beta_value_agent_init=learn_beta_agent_init,
                              client_gender=self.client_gender,agent_gender=self.agent_gender,
                              agent_rough=self.roughening_noise,client_rough=self.roughening_noise,use_pomcp=self.use_pomcp,
                              init_turn=self.initial_turn,numcact=self.numcact,numdact=self.numdact,obsres=self.obsres,
                              actres=self.actres,pomcp_timeout=self.timeout)

        print 10*"-","learning agent parameters: "
        learn_agent.print_params()
        print "learner init tau: ",learn_tau_init
        print "learner prop init: ",learn_prop_init
        print "learner beta client init: ",learn_beta_client_init
        print "learner beta agent init: ",learn_beta_agent_init

        #the following two initialisation calls should be inside the Agent constructor to keep things cleaner
        learn_avgs=learn_agent.initialise_array(learn_tau_init,learn_prop_init,learn_initx)


        #To plot initial data
        if (None != self.plotter):
            #to send the initial sentiments to the plotter
            learn_agent.sendSamplesToPlotter(learn_agent.samples,self.plotter,eTurn.learner)
            self.plotter.plot()


        print "learner average sentiments (f): "
        learn_avgs.print_val()



        done = False
        iter=0
        while not done:

            print 10*"-","iter ",iter,80*"-"

            learn_avgs = learn_agent.getAverageState()

            print "agent state is: "
            learn_avgs.print_val()

            #this always works here, but is only needed to avoid asking the user too many questions
            #and to figure out the observation
            learn_turn=learn_avgs.get_turn()
            print learn_turn

            observ=[]

            #if use_pomcp or learn_turn=="agent":
            #get the next action for the agent - may be a null action if it is the client turn
            (learn_aab,learn_paab)=learn_agent.get_next_action(learn_avgs,exploreTree=True)

            if learn_turn=="agent":
                aact=findNearestBehaviour(learn_aab,fbehaviours_agent)
                print "suggested action for the agent is :",learn_aab,"\n  closest label is: ",aact

            if self.useEmotionalAgent:
                agentEmotion = learn_agent.expectedEmotion("agent")
                clientEmotion = learn_agent.expectedEmotion("client")
                clientEmotionLabel = learn_agent.findNearestEmotion(clientEmotion)
                print "agent is feeling: ",agentEmotion," which is : ",learn_agent.findNearestEmotion(agentEmotion)
                print "agent thinks client is feeling: ",clientEmotion," which is: ",clientEmotionLabel

            if learn_turn=="agent":
                #we only want to ask the user for an action if it is his turn,
                #although this could be relaxed to allow the agent to barge in
                #this will be relevant if the turn is non-deterministic, in which case there
                #may be some samples for each turn value, and we may want an action to take??
                learn_aab=ask_client(fbehaviours_agent,aact,learn_aab,learn_turn)
                print "agent does action :",learn_aab,"\n"
                learn_observ=[]
                learn_eobserv=[]
            else:
                #first, see what the agent would predict and suggest this to the client
                #this can be removed in a true interactive setting, so this is only here so we can see what is going on
                (client_aab,client_paab)=learn_agent.get_default_predicted_action(learn_avgs)
                aact=findNearestBehaviours(client_aab,fbehaviours_agent,10)
                print "agent advises the following action :",client_aab,"\n  closest labels are: ", [re.sub(r"_"," ",i.strip()) for i in aact]

                #now, this is where the client actually decides what to do, possibly looking at the suggested labels from the agent
                #we use fbehaviours_agent here (for agent gender) as it is the agent who is perceiving this
                learn_observ=ask_client(fbehaviours_agent,aact[0],client_aab,learn_turn)
                #should be to get a default (null)  action from the agent
                #learn_aab=[0.0,0.0,0.0]
                print "client action: ",learn_observ

                if self.useEmotionalAgent:
                    learn_eobserv = ask_client(learn_agent.emotdict,clientEmotionLabel,clientEmotion,learn_turn,"emotion")

            #we may be done if the user has killed the interaction
            if learn_turn=="client" and learn_observ==[]:
                done = True
            elif learn_turn=="agent" and learn_aab==[]:
                done = True
            else:
                #agent gets to observe the turn each time
                learn_xobserv=[State.turnnames.index(invert_turn(learn_turn))]

                #the main SMC update step
                if self.useEmotionalAgent:
                    learn_avgs=learn_agent.propagate_forward(learn_aab,learn_observ,learn_xobserv,learn_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.learner,eobserv=learn_eobserv)
                else:
                    learn_avgs=learn_agent.propagate_forward(learn_aab,learn_observ,learn_xobserv,learn_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.learner)

                #To plot data
                if (None != self.plotter):
                    self.plotter.plot()


                #I think these should be based on fundamentals, not transients
                (aid,cid)=learn_agent.get_avg_ids(learn_avgs.f)
                print "agent thinks it is most likely a: ",aid
                print "agent thinks the client is most likely a: ",cid

                if self.get_full_id_rate>0 and (iter+1)%self.get_full_id_rate==0:
                    (cnt_ags,cnt_cls)=learn_agent.get_all_ids()
                    print "agent thinks of itself as (full distribution): "
                    print cnt_ags[0:10]
                    print "agent thinks of the client as (full distribution): "
                    print cnt_cls[0:10]

                if self.useEmotionalAgent:
                    agentEmotion = learn_agent.expectedEmotion("agent")
                    clientEmotion = learn_agent.expectedEmotion("client")
                    clientEmotionLabel = learn_agent.findNearestEmotion(clientEmotion)
                    print "agent is feeling: ",agentEmotion," which is : ",learn_agent.findNearestEmotion(agentEmotion)
                    print "agent thinks client is feeling: ",clientEmotion," which is: ",clientEmotionLabel

                iter += 1
            print "current deflection of averages: ",learn_agent.deflection_avg

            learn_d=learn_agent.compute_deflection()
            print "current deflection (agent's perspective): ",learn_d

        if (None != self.plotter and None != self.plotter.m_PlotFrame):
            self.plotter.m_PlotFrame.Close()

def main(argv):
    plot = False
    oBayesactInteractive = cBayesactInteractive(argv, plotter=None)

    # A hack, should probably be fixed later
    helpstring="Bayesact simulator (2 agents) usage:\n bayesactsim.py\n\t -t <number of trials (default 20)>\n\t -x <number of experiments per trial (default 10)>\n\t -n <number of samples (default 1000)>\n\t -c <client knowledge (0,1,2) (default 2)>\n\t -a <agent knowledge (0,1,2) (Default 0)>\n\t -u (if specified - do uniform draws)\n\t -d <max horizon - default 50>\n\t -r <roughening noise: default n^(-1/3) - to use no roughening ('full' method), specify 0>\n\t -e <environment noise (default 0.0)>\n\t -g <gamma_value (default 1.0)>\n\t -i <agent id label: default randomly chosen>\n\t -j <client id label: default randomly chosen>\n\t -k <agent gender (default: male) - only works if agent_id is specified with -i>\n\t -l (client gender (default: male) only works if client_id is specified with -j>\n\t -m <gammae_value (default 0.0 or emotions not used)>\n\t"

    try:
        opts, args = getopt.getopt(argv[1:],"huvon:t:x:a:c:d:r:e:g:i:j:m:k:l:",["help","n=","t=","x=","c=","a=","u=","d=","r=","e=","g=","i=","j=","k=","l="])
    except getopt.GetoptError:
        print helpstring
        sys.exit(2)

    for opt, arg in opts:
        if "-o" == opt:
            plot = True

    if (False == plot):
        oBayesactInteractive.startBayesactInteractive()
    else:
        from cPlotBayesactThread import cPlotBayesactThread
        plotter = cPlotBayesactThread()
        plotPanel = plotter.initFrame()
        plotter.initPlotBayesactSim(plotPanel)
        oBayesactInteractive.plotter = plotter

        bayesactSimThread = threading.Thread(target=oBayesactInteractive.startBayesactInteractive)
        plotter.setThread(bayesactSimThread)
        plotter.startApp()


if __name__ == "__main__":
    main(sys.argv)
