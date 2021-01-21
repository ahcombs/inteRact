import sys
import threading
sys.path.append("../")
from bayesact import *
import cProfile
from cEnum import eTurn, eGui
from cBayesactSimGui import cBayesactSimGuiPanel


#simple threading class so we can do two things at once
class propagateForwardThread (threading.Thread):
    def __init__(self, tAgent, aab,observ,xobserv=[],paab=None,verb=False,plotter=None,agent=eTurn.learner):
        threading.Thread.__init__(self)
        self.tAgent = tAgent
        self.result = None
        self.aab = aab
        self.observ = observ
        self.xobserv = xobserv
        self.paab = paab
        self.verb = verb
        self.plotter = plotter
        self.agent = agent
    def run(self):
        self.result = self.tAgent.propagate_forward(self.aab, self.observ, self.xobserv, self.paab, self.verb, self.plotter, self.agent)


class cBayesactSim(object):
    def __init__(self):
        #NP.set_printoptions(precision=5)
        #NP.set_printoptions(suppress=True)
        NP.set_printoptions(linewidth=10000)

        #-----------------------------------------------------------------------------------------------------------------------------

        #get some key parameters from the command line
        self.num_samples=1000

        #agent knowledge of client id:
        #0 : nothing
        #1 : one of a selection of  num_confusers+1 randoms
        #2 : exactly
        #3 : same as 0 but also agent does not know its own id
        self.agent_knowledge=0
        #client knowledge of agent id:
        self.client_knowledge=0

        self.agent_id_label=""
        self.client_id_label=""


        self.agent_gender="male"
        self.client_gender="male"


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


        #used only for label extraction if the user enters a label they want to use for simulations
        self.identities_filename="fidentities.dat"

        self.behaviours_filename="fbehaviours.dat"

        #if>0, add a small amount of roughening noise to the client identities
        self.roughening_noise=-1.0


        #environment noise: this will corrupt the transmission of the actions with zero-mean Gaussian noise with this variance
        self.env_noise=0.0

        self.get_full_id_rate=20

        self.learn_init_turn="agent"
        self.simul_init_turn="client"

        self.verbose=False

        if self.roughening_noise<0.0:
            #default
            self.roughening_noise=self.num_samples**(-1.0/3.0)





        #-----------------------------------------------------------------------------------------------------------------------------
        #simulation start
        #-----------------------------------------------------------------------------------------------------------------------------

        self.num_agent_confusers=3
        self.num_client_confusers=3



        #to roughen, not use proposal  - for any unknown identity
        self.learn_agent_rough=0.0
        self.learn_client_rough=0.0

        if (self.agent_knowledge==0 or self.agent_knowledge==3) and self.roughening_noise>0:
            self.learn_client_rough=self.roughening_noise
            if self.agent_knowledge==3:
                self.learn_agent_rough=self.roughening_noise

        self.simul_agent_rough=0.0
        self.simul_client_rough=0.0
        if (self.client_knowledge==0 or self.client_knowledge==3) and self.roughening_noise>0:
            self.simul_client_rough=self.roughening_noise
            if self.client_knowledge==3:
                self.simul_agent_rough=self.roughening_noise



        self.simul_verbose=self.verbose
        self.learn_verbose=self.verbose

        self.meanlearndefl =[]
        self.meansimuldefl = []
        self.meanlearnddvo = []
        self.meansimulddvo = []


        #the mean and covariance of IDs for male agents as taken from the databases
        #should do this automatically in python based on actual genders of client/agent....
        self.mean_ids=NP.array([0.40760,0.40548,0.45564])
        self.cov_ids=NP.array([[2.10735,1.01121, 0.48442],[1.01121,1.22836,0.55593],[0.48442,0.55593,0.77040]])


        #------------------------------------------------------------------------------
        # Rand seed

        # Does this do anything?
        self.randseeds=[373044980,27171222,156288614]


        #this was moved from below the client_id agent_id generation
        #on March 22nd at 11:52am
        #for repeatability
        self.rseed = NP.random.randint(0,382948932)
        #rseed=randseeds[trial]
        #rseed = 373044980  #lady and shoplifter
        #rseed = 156288614  #hero and insider
        #rseed=60890393
        #rseed=120534679

        #------------------------------------------------------------------------------

        #to make it consistent across different trials
        #NP.random.seed(1)

        # alpha beta and gamma values
        self.client_alpha_value = 0.1
        self.client_beta_value_of_client = 0.005
        self.client_beta_value_of_agent = 0.005
        self.client_gamma_value = 1.0

        self.agent_alpha_value = 0.1
        self.agent_beta_value_of_client = 0.005
        self.agent_beta_value_of_agent = 0.005
        self.agent_gamma_value = 1.0


        # for interacting
        self.fbehaviours_agent = None
        self.fbehaviours_client = None

        '''
        self.agent_mean_ids = None
        self.agent_cov_ids = None
        self.client_mean_ids = None
        self.client_cov_ids = None
        '''

        # from trials block
        self.client_id = None
        self.agent_id = None

        self.learn_tau_init = None
        self.learn_prop_init = None
        self.learn_beta_client_init = None
        self.learn_beta_agent_init = None

        self.simul_tau_init = None
        self.simul_prop_init = None
        self.simul_beta_client_init = None
        self.simul_beta_agent_init = None

        self.learn_initx = None
        self.simul_initx = None

        self.simul_agent = None
        self.learn_agent = None

        self.learn_ddvo = None
        self.learn_defl = None
        self.simul_ddvo = None
        self.simul_defl = None


        # from experiments block
        self.learn_avgs = None
        self.simul_avgs = None
        self.learn_turn = None
        self.simul_turn = None


        # from horizon block
        self.learn_d = None
        self.simul_d = None
        self.total_iterations = 0

        # plotting and for gui
        self.plotter = None

        # The sim_gui is an instance of cBayesactSimGui
        self.sim_gui = None

        # The interactive_gui is an instance of cBayesactInteractiveGui
        self.interactive_gui = None

        self.thread_event = None
        self.waiting = True
        self.terminate_flag = False

        # for running instructions
        self.update_environment_noise           = False

        self.update_client_alpha                = False
        self.update_client_beta_of_client       = False
        self.update_client_beta_of_agent        = False
        self.update_client_gamma                = False

        self.update_agent_alpha                 = False
        self.update_agent_beta_value_of_client  = False
        self.update_agent_beta_value_of_agent   = False
        self.update_agent_gamma                 = False


        # This is for the simulator to specify how many times to step
        self.update_num_steps                   = False
        self.step_bayesact_sim                  = False

        # Special case where user wants to press the step command first instead of the start command
        self.initial_step_bayesact_sim          = False

        self.num_steps                          = 1

        self.evaluate_interaction               = False
        self.user_input_action                  = ""


        self.next_learn_aab                     = []
        self.next_learn_paab                    = []

        self.next_simul_aab                     = []
        self.next_simul_paab                    = []

        self.last_action_agent                  = ""
        self.last_action_client                 = ""


    def startThread(self):
        self.waiting = False

        print "random seed is : ", self.rseed
        print "num samples: ",self.num_samples
        print "client knowledge: ",self.client_knowledge
        print "agent knowledge: ",self.agent_knowledge
        print "roughening noise: ",self.roughening_noise
        print "environment noise: ",self.env_noise
        print "client gamma value: ",self.client_gamma_value
        print "agent gamma value: ",self.agent_gamma_value
        print "client id label: ",self.client_id_label
        print "agent id label: ",self.agent_id_label
        print "agent gender: ",self.agent_gender
        print "client gender: ",self.client_gender

        self.initBayesactSim()

        if (self.initial_step_bayesact_sim):
            self.stepBayesactSim(self.num_steps)
            self.initial_step_bayesact_sim = False

        self.mainLoop()


    def mainLoop(self):
        while(not(self.terminate_flag)):
            # For some reason, while a thread is waiting, the program will say it is leaking memory sometimes
            self.waiting = True
            # Pause and await instruction
            self.thread_event.clear()
            self.thread_event.wait()
            self.runInstructions()
            self.waiting = False
            #self.plotter.plot()

        self.plotter.clearPlots()
        self.sim_gui.m_PreviousIterationsTextBox.SetValue(str(self.total_iterations))
        self.interactive_gui.m_PreviousIterationsTextBox.SetValue(str(self.total_iterations))
        self.total_iterations = 0
        self.sim_gui.m_CurrentIterationsTextBox.SetValue(str(self.total_iterations))
        self.interactive_gui.m_CurrentIterationsTextBox.SetValue(str(self.total_iterations))


    def runInstructions(self):
        if (self.step_bayesact_sim):
            if (self.update_num_steps):
                self.num_steps = int(self.sim_gui.m_NumberOfStepsTextBox.GetValue())
                self.update_num_steps = False

            self.stepBayesactSim(self.num_steps)
            self.step_bayesact_sim = False
            return

        if (self.evaluate_interaction):
            self.user_input_action = self.interactive_gui.m_ActionTextBox.GetValue()
            self.evaluateInteraction()
            self.evaluate_interaction = False


    def stepBayesactSim(self, iNumSteps):
        for i in range(iNumSteps):
            self.checkParams()
            self.evaluateStep()

            if (self.terminate_flag):
                return


    def checkParams(self):
        recompute_client = False
        recompute_agent = False

        if (self.update_environment_noise):
            self.env_noise = float(self.sim_gui.m_EnvironmentNoiseTextBox.GetValue())
            self.update_environment_noise = False

        if (self.update_client_alpha):
            self.client_alpha_value = float(self.sim_gui.m_ClientAlphaTextBox.GetValue())
            self.simul_agent.alpha_value = self.client_alpha_value
            recompute_client = True
            self.update_client_alpha = False

        if (self.update_client_beta_of_client):
            self.client_beta_value_of_client = float(self.sim_gui.m_ClientBetaOfClientTextBox.GetValue())
            self.simul_agent.beta_value_client = self.client_beta_value_of_client
            recompute_client = True
            self.update_client_beta_of_client = False

        if (self.update_client_beta_of_agent):
            self.client_beta_value_of_agent = float(self.sim_gui.m_ClientBetaOfAgentTextBox.GetValue())
            self.simul_agent.beta_value_agent = self.client_beta_value_of_agent
            recompute_client = True
            self.update_client_beta_of_agent = False

        if (self.update_client_gamma):
            self.client_gamma_value = float(self.sim_gui.m_AgentGammaTextBox.GetValue())
            self.simul_agent.gamma_value = self.client_gamma_value
            recompute_client = True
            self.update_client_gamma = False

        if (self.update_agent_alpha):
            self.agent_alpha_value = float(self.sim_gui.m_AgentAlphaTextBox.GetValue())
            self.learn_agent.alpha_value = self.agent_alpha_value
            recompute_agent = True
            self.update_agent_alpha = False

        if (self.update_agent_beta_value_of_client):
            self.agent_beta_value_of_client = float(self.sim_gui.m_AgentBetaOfClientTextBox.GetValue())
            self.learn_agent.beta_value_client = self.agent_beta_value_of_client
            recompute_agent = True
            self.update_agent_beta_value_of_client = False

        if (self.update_agent_beta_value_of_agent):
            self.learn_beta_value_of_agent = float(self.sim_gui.m_AgentBetaOfAgentTextBox.GetValue())
            self.simul_agent.beta_value_agent = self.agent_beta_value_of_agent
            recompute_agent = True
            self.update_agent_beta_value_of_agent = False

        if (self.update_agent_gamma):
            self.agent_gamma_value = float(self.sim_gui.m_AgentGammaTextBox.GetValue())
            self.learn_agent.gamma_value = self.agent_gamma_value
            recompute_agent = True
            self.update_agent_gamma = False


        # To be used after updating alpha, beta, and gamma values
        if (recompute_client):
            self.simul_agent.init_covariances()
            recompute_client = False

        if (recompute_agent):
            self.learn_agent.init_covariances()
            recompute_agent = False


    def printResults(self):
        print 10*"+"," RESULTS",10*"+"
        print "final deflection (learner): ",self.learn_d
        self.learn_defl.append(self.learn_d)
        print "final learned client id: "
        print self.learn_avgs.f[6:9]
        print "actual client id: "
        print self.client_id.transpose()
        dvo= self.client_id.transpose()-self.learn_avgs.f[6:9]
        #changed from sqrt(dot(...)) here on April 4th, 2013
        self.learn_ddvo.append(NP.dot(dvo,dvo.transpose()))
        print "sum of squared differences between agent learned and client true id:"
        print self.learn_ddvo[-1]
        print "final deflection (simulator): ",self.simul_d
        self.simul_defl.append(self.simul_d)
        print "final learned agent id: "
        print self.simul_avgs.f[6:9]
        print "actual agent id: "
        print self.agent_id.transpose()
        dvo = self.agent_id.transpose()-self.simul_avgs.f[6:9]
        #changed from sqrt(dot(...)) here on April 4th, 2013
        self.simul_ddvo.append(NP.dot(dvo,dvo.transpose()))
        print "sum of squared differences between simul learned and agent true id:"
        print self.simul_ddvo[-1]
        print 30*"+"

        self.meanlearndefl.append(NP.mean(self.learn_defl))
        self.meansimuldefl.append(NP.mean(self.simul_defl))
        self.meanlearnddvo.append(NP.mean(self.learn_ddvo))
        self.meansimulddvo.append(NP.mean(self.simul_ddvo))

        stdlearndefl = NP.std(self.meanlearndefl)
        stdsimuldefl = NP.std(self.meansimuldefl)
        stdlearnddvo = NP.std(self.meanlearnddvo)
        stdsimulddvo = NP.std(self.meansimulddvo)

        print 10*"*","OVERALL RESULTS",10*"*"
        print "mean deflection (learner): ",NP.mean(self.meanlearndefl)," +/- ",stdlearndefl
        print "mean deflection (simulator): ",NP.mean(self.meansimuldefl)," +/- ",stdsimuldefl
        print "mean difference in id sentiments (learner): ",NP.mean(self.meanlearnddvo)," +/- ",stdlearnddvo
        print "mean differnece in id sentiments (simulator): ",NP.mean(self.meansimulddvo)," +/- ",stdsimulddvo
        print 30*"*"


    def mimicInteract(self):
        self.agent_beta_of_agent_init=0.000001
        self.agent_beta_of_client_init=0.000001
        self.agent_beta_of_agent=0.00001
        self.agent_beta_of_client=0.00001

        self.client_beta_of_agent_init=0.000001
        self.client_beta_of_client_init=0.000001
        self.client_beta_of_agent=0.00001
        self.client_beta_of_client=0.00001

        self.agent_knowledge=2
        self.num_samples=10000
        self.roughening_noise=0.0


    def initBayesactSim(self):

        self.fbehaviours_agent=readSentiments(self.behaviours_filename,self.agent_gender)
        self.fbehaviours_client=readSentiments(self.behaviours_filename,self.client_gender)

        NP.random.seed(self.rseed)

        # In order to copy bayesactsim, this method of drawing the distribution over ids is not used
        '''
        (self.agent_mean_ids,self.agent_cov_ids)=getIdentityStats(self.identities_filename,self.agent_gender)
        (self.client_mean_ids,self.client_cov_ids)=getIdentityStats(self.identities_filename,self.client_gender)

        #the actual (true) ids drawn from the distribution over ids
        if self.client_id_label=="":
            self.client_id = NP.asarray([NP.random.multivariate_normal(self.client_mean_ids,self.client_cov_ids)]).transpose()
        else:
            self.client_id = NP.asarray([getIdentity(self.identities_filename,self.client_id_label,self.client_gender)]).transpose()
        if self.agent_id_label=="":
            self.agent_id =  NP.asarray([NP.random.multivariate_normal(self.agent_mean_ids,self.agent_cov_ids)]).transpose()
        else:
            self.agent_id = NP.asarray([getIdentity(self.identities_filename,self.agent_id_label,self.agent_gender)]).transpose()

        # The agent
        (self.learn_tau_init,self.learn_prop_init,self.learn_beta_client_init,self.learn_beta_agent_init)=init_id(self.agent_knowledge, self.agent_id, self.client_id, self.client_mean_ids, self.num_agent_confusers)
        # The client
        (self.simul_tau_init,self.simul_prop_init,self.simul_beta_client_init,self.simul_beta_agent_init)=init_id(self.client_knowledge, self.client_id, self.agent_id, self.agent_mean_ids, self.num_client_confusers)
        '''

        #the actual (true) ids drawn from the distribution over ids
        if self.client_id_label=="":
            self.client_id = NP.asarray([NP.random.multivariate_normal(self.mean_ids,self.cov_ids)]).transpose()
        else:
            self.client_id = NP.asarray([getIdentity(self.identities_filename,self.client_id_label,self.client_gender)]).transpose()
        if self.agent_id_label=="":
            self.agent_id =  NP.asarray([NP.random.multivariate_normal(self.mean_ids,self.cov_ids)]).transpose()
        else:
            self.agent_id = NP.asarray([getIdentity(self.identities_filename,self.agent_id_label,self.agent_gender)]).transpose()


        (self.learn_tau_init,self.learn_prop_init,self.learn_beta_client_init,self.learn_beta_agent_init)=init_id(self.agent_knowledge, self.agent_id, self.client_id, self.mean_ids, self.num_agent_confusers)
        (self.simul_tau_init,self.simul_prop_init,self.simul_beta_client_init,self.simul_beta_agent_init)=init_id(self.client_knowledge, self.client_id, self.agent_id, self.mean_ids, self.num_client_confusers)

        print "agent id: ",self.agent_id
        print "client id: ",self.client_id

        self.learn_initx=self.learn_init_turn
        self.simul_initx=self.simul_init_turn

        if self.mimic_interact:
            self.mimicInteract()


        # The agent
        self.learn_agent=Agent(N=self.num_samples, alpha_value=self.agent_alpha_value,
                               gamma_value=self.agent_gamma_value ,beta_value_agent=self.agent_beta_value_of_agent ,beta_value_client=self.agent_beta_value_of_client,
                               beta_value_client_init=self.learn_beta_client_init, beta_value_agent_init=self.learn_beta_agent_init,
                               client_gender=self.client_gender, agent_gender=self.agent_gender,
                               agent_rough=self.learn_agent_rough,client_rough=self.learn_client_rough,identities_file=self.identities_filename, use_pomcp=self.use_pomcp,
                               init_turn=self.learn_init_turn, numcact=self.numcact, numdact=self.numdact, obsres=self.obsres, actres=self.actres, pomcp_timeout=self.timeout)

        # The client
        self.simul_agent=Agent(N=self.num_samples, alpha_value=self.client_alpha_value,
                               gamma_value=self.client_gamma_value, beta_value_agent=self.client_beta_value_of_agent, beta_value_client=self.client_beta_value_of_client,
                               beta_value_client_init=self.simul_beta_client_init, beta_value_agent_init=self.simul_beta_agent_init,
                               client_gender=self.client_gender, agent_gender=self.agent_gender,
                               agent_rough=self.simul_agent_rough, client_rough=self.simul_client_rough, identities_file=self.identities_filename, use_pomcp=self.use_pomcp,
                               init_turn=self.simul_init_turn, numcact=self.numcact, numdact=self.numdact, obsres=self.obsres, actres=self.actres, pomcp_timeout=self.timeout)


        self.simul_agent.print_params()
        print "simulator init tau: ",self.simul_tau_init
        print "simulator prop init: ",self.simul_prop_init
        print "simulator beta client init: ",self.simul_beta_client_init
        print "simulator beta agent init: ",self.simul_beta_agent_init

        print 10*"-","learning agent parameters: "
        self.learn_agent.print_params()
        print "learner init tau: ",self.learn_tau_init
        print "learner prop init: ",self.learn_prop_init
        print "learner beta client init: ",self.learn_beta_client_init
        print "learner beta agent init: ",self.learn_beta_agent_init

        self.learn_ddvo=[]
        self.learn_defl=[]
        self.simul_ddvo=[]
        self.simul_defl=[]

        self.learn_avgs=self.learn_agent.initialise_array(self.learn_tau_init,self.learn_prop_init,self.learn_initx)
        self.simul_avgs=self.simul_agent.initialise_array(self.simul_tau_init,self.simul_prop_init,self.simul_initx)

        print "simulator average: "
        self.simul_avgs.print_val()

        print "learner average f: "
        self.learn_avgs.print_val()

        self.learn_turn="agent"
        self.simul_turn="client"


        #To plot initial data
        #to send the initial sentiments to the plotter
        self.learn_agent.sendSamplesToPlotter(self.learn_agent.samples,self.plotter,eTurn.learner)
        self.simul_agent.sendSamplesToPlotter(self.simul_agent.samples,self.plotter,eTurn.simulator)
        self.plotter.plot()

        (self.next_learn_aab,self.next_learn_paab)=self.learn_agent.get_next_action(self.learn_avgs,exploreTree=True)
        (self.next_simul_aab,self.next_simul_paab)=self.simul_agent.get_next_action(self.simul_avgs,exploreTree=True)

        self.interactive_gui.m_CurrentTurnTextBox.SetValue(self.learn_turn)
        self.populateSuggestedActions(findNearestBehaviours(self.next_learn_aab, self.fbehaviours_agent, 10))

        self.terminate_flag = False
        self.total_iterations=0


    def evaluateStep(self):
        print "learner turn: ",self.learn_turn
        print "simulator turn: ",self.simul_turn

        observ=[]

        print 10*"-d","iteration number ",self.total_iterations,80*"-"

        # To get the next action and propagate_forward on it
        # ####################################################################

        print "agent action/client observ: ",self.next_learn_aab
        simul_observ=self.next_learn_aab

        print "client action/agent observ: ",self.next_simul_aab
        learn_observ=self.next_simul_aab

        #add environmental noise here if it is being used
        if self.env_noise>0.0:
            learn_observ=map(lambda fv: NP.random.normal(fv,self.env_noise), learn_observ)
            simul_observ=map(lambda fv: NP.random.normal(fv,self.env_noise), simul_observ)


        print "learn observ: ",learn_observ
        print "simul observ: ",simul_observ


        learn_xobserv=[State.turnnames.index(invert_turn(self.learn_turn))]
        simul_xobserv=[State.turnnames.index(invert_turn(self.simul_turn))]


        #learn_avgs=cProfile.run('learn_agent.propagate_forward(learn_turn,learn_aab,learn_observ,verb=learn_verbose)')
        # propagate_forward sends the unweigted data set to the plotter, hence the reason why there isn't a function there isn't an explicit sendSamplesToPlotter function here
        self.learn_avgs=self.learn_agent.propagate_forward(self.next_learn_aab,learn_observ,learn_xobserv,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.learner)
        self.simul_avgs=self.simul_agent.propagate_forward(self.next_simul_aab,simul_observ,simul_xobserv,verb=self.simul_verbose,plotter=self.plotter,agent=eTurn.simulator)

        # The multithreaded code here is horrendously slow, I will leave it here for future reference
        '''
        learnThread = propagateForwardThread(self.learn_agent,self.next_learn_aab,learn_observ,learn_xobserv,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.learner)
        simulThread = propagateForwardThread(self.simul_agent,self.next_simul_aab,simul_observ,simul_xobserv,verb=self.simul_verbose,plotter=self.plotter,agent=eTurn.simulator)
        learnThread.start()
        simulThread.start()

        while learnThread.isAlive() or simulThread.isAlive():
            pass

        self.learn_avgs = learnThread.result
        self.simul_avgs = simulThread.result
        '''

        # ####################################################################

        print "learner f is: "
        self.learn_avgs.print_val()

        print "simulator f is: "
        self.simul_avgs.print_val()


        # To get the nearest ids for the agent or client
        #I think these should be based on fundamentals, not transients
        (aid,cid)=self.learn_agent.get_avg_ids(self.learn_avgs.f)
        print "learner agent id:",aid
        print "learner client id:",cid

        (aid,cid)=self.simul_agent.get_avg_ids(self.simul_avgs.f)
        print "simul agent id:",aid
        print "simul client id:",cid

        '''
        if self.get_full_id_rate>0 and (self.total_iterations+1)%self.get_full_id_rate==0:
            (cnt_ags,cnt_cls)=self.learn_agent.get_all_ids()
            print "top ten ids for agent (learner perspective):"
            print cnt_ags[0:10]
            print "top ten ids for client (learner perspective):"
            print cnt_cls[0:10]
            (cnt_ags,cnt_cls)=simul_agent.get_all_ids()
            print "top ten ids for agent (simulator perspective):"
            print cnt_ags[0:10]
            print "top ten ids for client (simulator perspective):"
            print cnt_cls[0:10]
        '''

        print "current deflection of averages: ",self.learn_agent.deflection_avg

        self.learn_d=self.learn_agent.compute_deflection()
        print "current deflection (learner perspective): ",self.learn_d
        self.simul_d=self.simul_agent.compute_deflection()
        print "current deflection (simulator perspective): ",self.simul_d


        if self.learn_turn=="client":
            self.learn_turn="agent"
            self.simul_turn="client"
        else:
            self.learn_turn="client"
            self.simul_turn="agent"

        #To plot data
        self.plotter.plot()


        #In fact, both agents should update on every turn now, so
        #we should be able to always call get_next_action for both agents here?

        #like this:
        (self.next_learn_aab,self.next_learn_paab)=self.learn_agent.get_next_action(self.learn_avgs,exploreTree=True)
        (self.next_simul_aab,self.next_simul_paab)=self.simul_agent.get_next_action(self.simul_avgs,exploreTree=True)

        # We know who goes next, so populate the suggested actions
        self.interactive_gui.m_CurrentTurnTextBox.SetValue(self.learn_turn)
        if (self.learn_turn == "agent"):
            self.populateSuggestedActions(findNearestBehaviours(self.next_learn_aab, self.fbehaviours_agent, 10))
        else:
            self.populateSuggestedActions(findNearestBehaviours(self.next_simul_aab, self.fbehaviours_client, 10))

        self.total_iterations += 1
        self.sim_gui.m_CurrentIterationsTextBox.SetValue(str(self.total_iterations))
        self.interactive_gui.m_CurrentIterationsTextBox.SetValue(str(self.total_iterations))


    def is_array_of_floats(self,possarray):
        try:
            parr = [float(x) for x in possarray.split(',')]
        except ValueError:
            return False
        if len(parr)==3:
            return parr
        return False

    #a function to ask the client for an action to take and map it to an EPA value
    def ask_client(self,fbeh,sugg_act='',epaact=[],who="agent"):
        sst="action to take for "+who+" ('?' shows options): "

        sstnc="You can now either :\n"
        sstnc+="- pick an action (behaviour) and type in its label\n"
        sstnc+="- type 'quit' to stop the simulation\n"
        sstnc+="- type a comma separated list of three numbers to specify the action as E,P,A values\n"
        if not sugg_act=='':
            sstnc+="- hit 'enter' to take default action with label: "+sugg_act[0]+"\n"
        if not epaact=='':
            sstnc+="- type any digit (0-9) to take suggsted action: "+str(epaact)


        cact = self.user_input_action

        if cact=='quit':
            return []

        elif cact=='' and not sugg_act=='':
            cact=sugg_act[0]

        elif cact.isdigit() and not epaact==[]:
            return epaact

        elif re.sub(r"\s+","_",cact.strip()) in fbeh.keys():
            cact=re.sub(r"\s+","_",cact.strip())

        elif cact=="?":
            print sstnc
            return False

        elif self.is_array_of_floats(cact):
            return [float(x) for x in cact.split(',')]

        else:
            print "incorrect or not found, try again. '?' shows options"
            return False

        observ=map(lambda x: float(x), [fbeh[cact]["e"],fbeh[cact]["p"],fbeh[cact]["a"]])
        print "client said: ",cact
        return observ


    def populateSuggestedActions(self, iActions):
        if (isinstance(iActions, list)):
            self.interactive_gui.m_SuggestedActionsListBox.SetItems(iActions)
        else:
            self.interactive_gui.m_SuggestedActionsListBox.SetItems([iActions])


    # There is no environment noise, should it be added?
    def evaluateInteraction(self):
        print 10*"-","iter ",self.total_iterations,80*"-"

        self.simul_avgs = self.simul_agent.getAverageState()
        self.learn_avgs = self.learn_agent.getAverageState()


        print "agent state is: "
        self.learn_avgs.print_val()

        print "client state is: "
        self.simul_avgs.print_val()

        #this always works here, but is only needed to avoid asking the user too many questions
        #to figure out the observation
        #self.learn_turn = self.learn_avgs.get_turn()
        #self.simul_turn = self.simul_avgs.get_turn()
        # Note: I have assumed that the agents will take turns, so we do not have to constantly get turns


        # To get the next action and propagate_forward on it
        # ####################################################################
        aact = self.interactive_gui.m_SuggestedActionsListBox.GetItems()
        if self.learn_turn=="agent":
            learn_aab = self.ask_client(self.fbehaviours_client,aact,self.next_learn_aab,self.learn_turn)
            if (False == learn_aab):
                return

            simul_observ = learn_aab
            print "client action :",learn_aab, "\n"
            learn_observ=[]

        else:
            simul_aab = self.ask_client(self.fbehaviours_agent,aact,self.next_simul_aab,self.learn_turn)
            if (False == simul_aab):
                return

            learn_observ = simul_aab
            #should be to get a default (null)  action from the agent
            print "agent action: ", simul_aab, "\n"
            simul_observ = []



        # ####################################################################


        #we may be done if the user has killed the interaction
        if self.learn_turn=="client" and learn_observ==[]:
            self.terminate_flag = True

        elif self.learn_turn=="agent" and learn_aab==[]:
            self.terminate_flag = True
        else:
            #observe the turn each time
            learn_xobserv=[State.turnnames.index(invert_turn(self.learn_turn))]
            simul_xobserv=[State.turnnames.index(invert_turn(self.simul_turn))]


            #the main SMC update step
            if (self.learn_turn == "agent"):
                self.learn_avgs=self.learn_agent.propagate_forward(learn_aab,learn_observ,learn_xobserv,self.next_learn_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.learner)
                self.simul_avgs=self.simul_agent.propagate_forward(self.next_simul_aab,simul_observ,simul_xobserv,self.next_simul_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.simulator)

                '''
                learnThread = propagateForwardThread(self.learn_agent,learn_aab,learn_observ,learn_xobserv,self.next_learn_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.learner)
                simulThread = propagateForwardThread(self.simul_agent,simul_observ,simul_xobserv,self.next_simul_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.simulator)
                learnThread.start()
                simulThread.start()

                while learnThread.isAlive() or simulThread.isAlive():
                    pass

                self.learn_avgs = learnThread.result
                self.simul_avgs = simulThread.result
                '''

            else:
                self.learn_avgs=self.learn_agent.propagate_forward(self.next_learn_aab,learn_observ,learn_xobserv,self.next_learn_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.learner)
                self.simul_avgs=self.simul_agent.propagate_forward(simul_aab,simul_observ,simul_xobserv,self.next_simul_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.simulator)

                '''
                learnThread = propagateForwardThread(self.learn_agent,self.next_learn_aab,learn_observ,learn_xobserv,self.next_learn_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.learner)
                simulThread = propagateForwardThread(self.simul_agent,simul_aab,simul_observ,simul_xobserv,self.next_simul_paab,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.simulator)
                learnThread.start()
                simulThread.start()

                while learnThread.isAlive() or simulThread.isAlive():
                    pass

                self.learn_avgs = learnThread.result
                self.simul_avgs = simulThread.result
                '''

            #To plot data
            if (None != self.plotter):
                self.plotter.plot()


            #I think these should be based on fundamentals, not transients
            (aid,cid)=self.learn_agent.get_avg_ids(self.learn_avgs.f)
            print "(learner's perspective) agent thinks it is most likely a: ",aid
            print "(learner's perspective) agent thinks the client is most likely a: ",cid

            (aid,cid)=self.simul_agent.get_avg_ids(self.simul_avgs.f)
            print "(simulator's perspective) learn agent thinks it is most likely a: ",aid
            print "(simulator's perspective) learn agent thinks the client is most likely a: ",cid


            if self.get_full_id_rate>0 and (self.total_iterations+1)%self.get_full_id_rate==0:
                (cnt_ags,cnt_cls)=self.learn_agent.get_all_ids()
                print "(learner's perspective) agent thinks of itself as (full distribution): "
                print cnt_ags[0:10]
                print "(learner's perspective) agent thinks of the client as (full distribution): "
                print cnt_cls[0:10]
                (cnt_ags,cnt_cls)=self.simul_agent.get_all_ids()
                print "(simulator's perspective) agent thinks of itself as (full distribution): "
                print cnt_ags[0:10]
                print "(simulator's perspective) agent thinks of the client as (full distribution): "
                print cnt_cls[0:10]

            self.total_iterations += 1
            self.sim_gui.m_CurrentIterationsTextBox.SetValue(str(self.total_iterations))
            self.interactive_gui.m_CurrentIterationsTextBox.SetValue(str(self.total_iterations))

        print "current deflection of averages: ",self.learn_agent.deflection_avg

        self.learn_d=self.learn_agent.compute_deflection()
        print "current deflection (agent's perspective): ",self.learn_d
        self.simul_d=self.simul_agent.compute_deflection()
        print "current deflection (simulator perspective): ",self.simul_d


        # Instead of getting the turn, we will assume this is a purely interactive simulation and they will take turns
        if self.learn_turn=="client":
            self.learn_turn="agent"
            self.simul_turn="client"
        else:
            self.learn_turn="client"
            self.simul_turn="agent"

        #if use_pomcp or learn_turn=="agent":
        #get the next action for the agent - may be a null action if it is the client turn
        (self.next_learn_aab,self.next_learn_paab)=self.learn_agent.get_next_action(self.learn_avgs,exploreTree=True)
        (self.next_simul_aab,self.next_simul_paab)=self.simul_agent.get_next_action(self.simul_avgs,exploreTree=True)

        self.learn_turn = self.learn_avgs.get_turn()
        self.interactive_gui.m_CurrentTurnTextBox.SetValue(self.learn_turn)
        if self.learn_turn=="agent":
            #aact=findNearestBehaviour(learn_aab,self.fbehaviours_agent)
            aact=findNearestBehaviours(self.next_learn_aab,self.fbehaviours_agent,10)
            print "suggested action for the agent is :",self.next_learn_aab,"\n  closest label is: ",aact

            # Populate the suggested actions into the BayesactInteractiveGui
            self.populateSuggestedActions(aact)
        else:
            aact=findNearestBehaviours(self.next_simul_aab,self.fbehaviours_client,10)
            print "suggested action for the client is :",self.next_simul_aab,"\n  closest label is: ",aact

            # Populate the suggested actions into the BayesactInteractiveGui
            self.populateSuggestedActions(aact)
