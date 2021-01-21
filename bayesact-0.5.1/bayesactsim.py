
"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Simulator Test Code
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
import getopt
import cProfile
import sys
import threading
sys.path.append("./gui/")
from cEnum import eTurn


class cBayesactSim(object):
    def __init__(self, argv):

        #NP.set_printoptions(precision=5)
        #NP.set_printoptions(suppress=True)
        NP.set_printoptions(linewidth=10000)


        #get some key parameters from the command line
        self.num_samples=1000

        #for the multiple id simulations, this was agent_knowledge=4, client_knowledge=2

        #agent knowledge of client id:
        #0 : nothing
        #1 : one of a selection of  num_confusers+1 randoms
        #2 : exactly
        #3 : same as 0 but also agent does not know its own id
        #4 : like 0 or 2 but multiple ids allowed to make a multimodal id
        self.agent_knowledge=4
        #client knowledge of agent id:
        self.client_knowledge=0

        #proportion of agent ids
        self.aidprop = [0.5,0.5]

        #self.agent_id_label=["daughter","employer"] #["rapist","priest"]
        self.agent_id_label=["mother","manager"] 
        #self.agent_id_label="daughter"
        #self.client_id_label="employee"
        #self.agent_id_client_label="employer"
        self.client_id_label="assistant" #"victim" #Catholic"

        #the client estimate of agent id
        self.agent_id_client_label="daughter" #"rapist" #priest"

        self.agent_gender="female"
        self.client_gender="female"

        #used only for label extraction if the user enters a label they want to use for simulations
        self.identities_filename="fidentities.dat"

        self.behaviours_filenmae="fbehaviours.dat"   #not used

        #uniform draws - if true will draw client IDs uniformly over -4.3,4.3.
        #If false, drawn from Normal distribution of the client ids in the database
        self.uniform_draws=False

        self.gamma_value=0.1

        #if>0, add a small amount of roughening noise to the client identities
        self.roughening_noise=-1.0


        #environment noise: this will corrupt the transmission of the actions with zero-mean Gaussian noise with this variance
        self.env_noise=0.0

        self.max_horizon=50

        self.num_trials=20
        self.num_experiments=10

        self.get_full_id_rate=5

        self.learn_init_turn="agent"
        self.simul_init_turn="client"

        self.verbose=True

        self.helpstring="Bayesact simulator (2 agents) usage:\n bayesactsim.py\n\t -t <number of trials (default 20)>\n\t -x <number of experiments per trial (default 10)>\n\t -n <number of samples (default 1000)>\n\t -c <client knowledge (0,1,2) (default 2)>\n\t -a <agent knowledge (0,1,2) (Default 0)>\n\t -u (if specified - do uniform draws)\n\t -d <max horizon - default 50>\n\t -r <roughening noise: default n^(-1/3) - to use no roughening ('full' method), specify 0>\n\t -e <environment noise (default 0.0)>\n\t -g <gamma_value (default 1.0)>\n\t -i <agent id label: default randomly chosen>\n\t  -j <client id label: default randomly chosen>\n\t -k <agent gender (default: male) - only works if agent_id is specified with -i>\n\t -l (client gender (default: male) only works if client_id is specified with -j>"

        try:
            opts, args = getopt.getopt(argv[1:],"huvon:t:x:a:c:d:r:e:g:i:j:k:l:",["help","n=","t=","x=","c=","a=","u=","d=","r=","e=","g=","i=","j=","k=","l="])
        except getopt.GetoptError:
            print self.helpstring
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print self.helpstring
                sys.exit()
            elif opt == '-v':
                self.verbose = True
            elif opt in ("-n", "--numsamples"):
                self.num_samples = int(arg)
            elif opt in ("-t", "--numtrials"):
                self.num_trials = int(arg)
            elif opt in ("-x", "--numexperiments"):
                self.num_experiments = int(arg)
            elif opt in ("-c", "--clientknowledge"):
                self.client_knowledge = int(arg)
            elif opt in ("-a", "--agentknowledge"):
                self.agent_knowledge = int(arg)
            elif opt in ("-d", "--horizon"):
                self.max_horizon = int(arg)
            elif opt in ("-u", "--uniformdraws"):
                self.uniform_draws=True
            elif opt in ("-r", "--roughen"):
                self.roughening_noise=float(arg)
            elif opt in ("-e", "--enoise"):
                self.env_noise=float(arg)
            elif opt in ("-g", "--gamma"):
                self.gamma_value=float(arg)
            elif opt in ("-i", "--agentid"):
                self.agent_id_label=arg
            elif opt in ("-j", "--clientid"):
                self.client_id_label=arg
            elif opt in ("-k", "--agentgender"):
                self.agent_gender=arg
            elif opt in ("-l", "--clientgender"):
                self.client_gender=arg


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

        if (self.agent_knowledge==0 or self.agent_knowledge==3 or self.agent_knowledge==4) and self.roughening_noise>0:
            self.learn_client_rough=self.roughening_noise
            if self.agent_knowledge==3 or self.agent_knowledge==4:
                self.learn_agent_rough=self.roughening_noise

        self.simul_agent_rough=0.0
        self.simul_client_rough=0.0
        if (self.client_knowledge==0 or self.client_knowledge==3 or self.client_knowledge==4) and self.roughening_noise>0:
            self.simul_client_rough=self.roughening_noise
            if self.client_knowledge==3:
                self.simul_agent_rough=self.roughening_noise

        #self.learn_agent_rough = self.roughening_noise
        #self.simul_agent_rough = self.roughening_noise
        #self.simul_client_rough = self.roughening_noise
        #self.learn_client_rough = self.roughening_noise

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


        # Does this do anything?
        self.randseeds=[373044980,27171222,156288614]

        #to make it consistent across different trials
        #NP.random.seed(1)

        self.plotter = None
        self.threadEvent = None
        self.waiting = True
        self.terminateFlag = False



    def startBayesactSim(self):
        self.waiting = False

        print "num trials: ",self.num_trials
        print "num experiments: ",self.num_experiments
        print "num samples: ",self.num_samples
        print "client knowledge: ",self.client_knowledge
        print "agent knowledge: ",self.agent_knowledge
        print "uniform draws: ",self.uniform_draws
        print "max horizon: ",self.max_horizon
        print "roughening noise: ",self.roughening_noise
        print "environment noise: ",self.env_noise
        print "gamma value: ",self.gamma_value
        print "client id label: ",self.client_id_label
        print "agent id label: ",self.agent_id_label
        print "agent gender: ",self.agent_gender
        print "client gender: ",self.client_gender



        for trial in range(self.num_trials):

            #this was moved from below the client_id agent_id generation
            #on March 22nd at 11:52am
            #for repeatability
            rseed = NP.random.randint(0,382948932)
            #rseed = 285907485
            #rseed=randseeds[trial]
            #rseed = 373044980  #lady and shoplifter
            #rseed = 156288614  #hero and insider
            #rseed=60890393
            #rseed=120534679
            rseed = 23330197
            print "random seeed is : ",rseed

            NP.random.seed(rseed)

            #the actual (true) ids drawn from the distribution over ids
            if self.client_id_label=="":
                client_id = NP.asarray([NP.random.multivariate_normal(self.mean_ids,self.cov_ids)]).transpose()
            else:
                client_id = NP.asarray([getIdentity(self.identities_filename,self.client_id_label,self.client_gender)]).transpose()
            if self.agent_id_label=="":
                agent_id =  NP.asarray([NP.random.multivariate_normal(self.mean_ids,self.cov_ids)]).transpose()
            else:
                if type(self.agent_id_label)==type([]):
                    agent_id=[]
                    for aid in self.agent_id_label:
                        agent_id.append(NP.asarray([getIdentity(self.identities_filename,aid,self.agent_gender)]).transpose())
                else:
                    agent_id = NP.asarray([getIdentity(self.identities_filename,self.agent_id_label,self.agent_gender)]).transpose()

            if self.agent_id_client_label=="":
                agent_id_client =  NP.asarray([NP.random.multivariate_normal(self.mean_ids,self.cov_ids)]).transpose()
            else:
                if type(self.agent_id_client_label)==type([]):
                    agent_id_client=[]
                    for aid in self.agent_id_client_label:
                        agent_id_client.append(NP.asarray([getIdentity(self.identities_filename,aid,self.agent_gender)]).transpose())
                else:
                    agent_id_client = NP.asarray([getIdentity(self.identities_filename,self.agent_id_client_label,self.agent_gender)]).transpose()

            print "agent id: ",agent_id
            print "client id: ",client_id

            #these are uniformly distributed which will generate waaay too many extreme ids
            #client_id = NP.dot(NP.random.random([3,1]),8.6)-4.3
            #agent_id = NP.dot(NP.random.random([3,1]),8.6)-4.3
            #simulation agent knows client and agent ids (it is the client)
            (learn_tau_init,learn_prop_init,learn_beta_client_init,learn_beta_agent_init)=init_id(self.agent_knowledge,agent_id,client_id,self.mean_ids,self.num_agent_confusers,self.aidprop)
            (simul_tau_init,simul_prop_init,simul_beta_client_init,simul_beta_agent_init)=init_id(self.client_knowledge,client_id,agent_id_client,self.mean_ids,self.num_client_confusers)

            learn_initx=self.learn_init_turn
            simul_initx=self.simul_init_turn

            #this version has the agents more flexible about their identities....
            #simul_agent=TutoringAgent(N=num_samples,alpha_value=1.0,initx=learn_initx,goalx=learn_goalx,gamma_value=1.0,beta_value_agent=0.05,beta_value_client=0.05,beta_value_client_init=simul_beta_init)
            #learn_agent=TutoringAgent(N=num_samples,alpha_value=1.0,initx=simul_initx,goalx=simul_goalx,gamma_value=1.0,beta_value_agent=0.05,beta_value_client=0.05,beta_value_client_init=learn_beta_init)
            simul_agent=Agent(N=self.num_samples,alpha_value=0.1,gamma_value=self.gamma_value,beta_value_agent=0.001,beta_value_client=0.001,beta_value_client_init=simul_beta_client_init,beta_value_agent_init=simul_beta_agent_init,agent_rough=self.simul_agent_rough,client_rough=self.simul_client_rough,identities_file=self.identities_filename,init_turn=self.learn_init_turn,agent_gender=self.client_gender)
            learn_agent=Agent(N=self.num_samples,alpha_value=0.1,gamma_value=self.gamma_value,beta_value_agent=0.01,beta_value_client=0.001,beta_value_client_init=learn_beta_client_init,beta_value_agent_init=learn_beta_agent_init,agent_rough=self.learn_agent_rough,client_rough=self.learn_client_rough,identities_file=self.identities_filename,init_turn=self.simul_init_turn,agent_gender=self.agent_gender)

            

            print 100*"$"
            print "trial number: ",trial
            print 100*"$"
            print 10*"-","simulation agent parameters: "
            simul_agent.print_params()
            print "simulator init tau: ",simul_tau_init
            print "simulator prop init: ",simul_prop_init
            print "simulator beta client init: ",simul_beta_client_init
            print "simulator beta agent init: ",simul_beta_agent_init

            print 10*"-","learning agent parameters: "
            learn_agent.print_params()
            print "learner init tau: ",learn_tau_init
            print "learner prop init: ",learn_prop_init
            print "learner beta client init: ",learn_beta_client_init
            print "learner beta agent init: ",learn_beta_agent_init

            learn_ddvo=[]
            learn_defl=[]
            simul_ddvo=[]
            simul_defl=[]


            for exper in range(self.num_experiments):
                print 100*"$"
                print "experiment number: ",exper
                print 100*"$"
                learn_avgs=learn_agent.initialise_array(learn_tau_init,learn_prop_init,learn_initx)
                simul_avgs=simul_agent.initialise_array(simul_tau_init,simul_prop_init,simul_initx)

                #To plot initial data
                if (None != self.plotter):
                    #to send the initial sentiments to the plotter
                    learn_agent.sendSamplesToPlotter(learn_agent.samples,self.plotter,eTurn.learner)
                    simul_agent.sendSamplesToPlotter(simul_agent.samples,self.plotter,eTurn.simulator)
                    self.plotter.plot()

                    if (None != self.threadEvent):
                        # For some reason, while a thread is waiting, the program will say it is leaking memory sometimes
                        self.waiting = True
                        self.threadEvent.wait()
                        self.waiting = False


                if (self.terminateFlag):
                    self.plotter.clearPlots()
                    self.waiting = True
                    return


                print "simulator average: "
                simul_avgs.print_val()

                print "learner average f: "
                learn_avgs.print_val()


                learn_turn="agent"
                simul_turn="client"

                for iter in range(self.max_horizon):
                    print "learner turn: ",learn_turn
                    print "simulator turn: ",simul_turn

                    observ=[]
                    print 10*"-d","iter ",iter,80*"-"
                    #In fact, both agents should update on every turn now, so
                    #we should be able to always call get_next_action for both agents here?

                    #like this:
                    (learn_aab,learn_paab)=learn_agent.get_next_action(learn_avgs)
                    print "agent action/client observ: ",learn_aab
                    simul_observ=learn_aab

                    (simul_aab,simul_paab)=simul_agent.get_next_action(simul_avgs)
                    print "client action/agent observ: ",simul_aab
                    learn_observ=simul_aab

                    #add environmental noise here if it is being used
                    if self.env_noise>0.0:
                        learn_observ=map(lambda fv: NP.random.normal(fv,self.env_noise), learn_observ)
                        simul_observ=map(lambda fv: NP.random.normal(fv,self.env_noise), simul_observ)

                    print "learn observ: ",learn_observ
                    print "simul observ: ",simul_observ

                    learn_xobserv=[State.turnnames.index(invert_turn(learn_turn))]
                    simul_xobserv=[State.turnnames.index(invert_turn(simul_turn))]



                    #learn_avgs=cProfile.run('learn_agent.propagate_forward(learn_turn,learn_aab,learn_observ,verb=learn_verbose)')
                    learn_avgs=learn_agent.propagate_forward(learn_aab,learn_observ,learn_xobserv,verb=self.learn_verbose,plotter=self.plotter,agent=eTurn.learner)
                    simul_avgs=simul_agent.propagate_forward(simul_aab,simul_observ,simul_xobserv,verb=self.simul_verbose,plotter=self.plotter,agent=eTurn.simulator)

                    print "learner f is: "
                    learn_avgs.print_val()

                    print "simulator f is: "
                    simul_avgs.print_val()

                    #I think these should be based on fundamentals, not transients
                    (aid,cid)=learn_agent.get_avg_ids(learn_avgs.f)
                    print "learner agent id:",aid
                    print "learner client id:",cid

                    (aid,cid)=simul_agent.get_avg_ids(simul_avgs.f)
                    print "simul agent id:",aid
                    print "simul client id:",cid


                    if self.get_full_id_rate>0 and (iter+1)%self.get_full_id_rate==0:
                        (cnt_ags,cnt_cls)=learn_agent.get_all_ids()
                        print "top ten ids for agent (learner perspective):"
                        print cnt_ags[0:10]
                        print "top ten ids for client (learner perspective):"
                        print cnt_cls[0:10]
                        (cnt_ags,cnt_cls)=simul_agent.get_all_ids()
                        print "top ten ids for agent (simulator perspective):"
                        print cnt_ags[0:10]
                        print "top ten ids for client (simulator perspective):"
                        print cnt_cls[0:10]

                    print "current deflection of averages: ",learn_agent.deflection_avg

                    learn_d=learn_agent.compute_deflection()
                    print "current deflection (learner perspective): ",learn_d
                    simul_d=simul_agent.compute_deflection()
                    print "current deflection (simulator perspective): ",simul_d

                    if learn_turn=="client":
                        learn_turn="agent"
                        simul_turn="client"
                    else:
                        learn_turn="client"
                        simul_turn="agent"


                    #To plot data
                    if (None != self.plotter):
                        self.plotter.plot()

                        if (None != self.threadEvent):
                            self.waiting = True
                            self.threadEvent.wait()
                            self.waiting = False


                    if (self.terminateFlag):
                        self.plotter.clearPlots()
                        self.waiting = True
                        return


                print 10*"+","EXPERIMENT ",exper," RESULTS",10*"+"
                print "final deflection (learner): ",learn_d
                learn_defl.append(learn_d)
                print "final learned client id: "
                print learn_avgs.f[6:9]
                print "actual client id: "
                print client_id.transpose()
                dvo= client_id.transpose()-learn_avgs.f[6:9]
                #changed from sqrt(dot(...)) here on April 4th, 2013
                learn_ddvo.append(NP.dot(dvo,dvo.transpose()))
                print "sum of squared differences between agent learned and client true id:"
                print learn_ddvo[-1]
                print "final deflection (simulator): ",simul_d
                simul_defl.append(simul_d)
                print "final learned agent id: "
                print simul_avgs.f[6:9]
                print "actual agent id: "
                print agent_id.transpose()
                dvo = agent_id.transpose()-simul_avgs.f[6:9]
                #changed from sqrt(dot(...)) here on April 4th, 2013
                simul_ddvo.append(NP.dot(dvo,dvo.transpose()))
                print "sum of squared differences between simul learned and agent true id:"
                print simul_ddvo[-1]
                print 30*"+"

            self.meanlearndefl.append(NP.mean(learn_defl))
            self.meansimuldefl.append(NP.mean(simul_defl))
            self.meanlearnddvo.append(NP.mean(learn_ddvo))
            self.meansimulddvo.append(NP.mean(simul_ddvo))

            stdlearndefl = NP.std(learn_defl)
            stdsimuldefl = NP.std(simul_defl)
            stdlearnddvo = NP.std(learn_ddvo)
            stdsimulddvo = NP.std(simul_ddvo)

            print 10*"*","TRIAL ",trial," RESULTS",10*"*"
            print "mean deflection (learner): ",self.meanlearndefl[-1]," +/- ",stdlearndefl
            print "mean deflection (simulator): ",self.meansimuldefl[-1]," +/- ",stdsimuldefl
            print "mean difference in id sentiments (learner): ",self.meanlearnddvo[-1]," +/- ",stdlearnddvo
            print "mean differnece in id sentiments (simulator): ",self.meansimulddvo[-1]," +/- ",stdsimulddvo
            print 30*"*"



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


        if (None != self.plotter and None != self.plotter.m_PlotFrame):
            self.plotter.m_PlotFrame.Close()


def main(argv):
    # This is required as plotting with wx does not allow wx.App.MainLoop to be started as a thread
    # I'm going to use the -o and -v option it can be changed anytime
    plot = False
    oBayesactSim = cBayesactSim(argv)

    # A hack, should probably be fixed later
    helpstring="Bayesact simulator (2 agents) usage:\n bayesactsim.py\n\t -t <number of trials (default 20)>\n\t -x <number of experiments per trial (default 10)>\n\t -n <number of samples (default 1000)>\n\t -c <client knowledge (0,1,2) (default 2)>\n\t -a <agent knowledge (0,1,2) (Default 0)>\n\t -u (if specified - do uniform draws)\n\t -d <max horizon - default 50>\n\t -r <roughening noise: default n^(-1/3) - to use no roughening ('full' method), specify 0>\n\t -e <environment noise (default 0.0)>\n\t -g <gamma_value (default 1.0)>\n\t -i <agent id label: default randomly chosen>\n\t  -j <client id label: default randomly chosen>\n\t -k <agent gender (default: male) - only works if agent_id is specified with -i>\n\t -l (client gender (default: male) only works if client_id is specified with -j>"

    try:
        opts, args = getopt.getopt(argv[1:],"huvon:t:x:a:c:d:r:e:g:i:j:k:l:",["help","n=","t=","x=","c=","a=","u=","d=","r=","e=","g=","i=","j=","k=","l="])
    except getopt.GetoptError:
        print helpstring
        sys.exit(2)

    for opt, arg in opts:
        if "-o" == opt:
            plot = True

    if (False == plot):
        oBayesactSim.startBayesactSim()
    else:
        from cPlotBayesactThread import cPlotBayesactThread
        plotter = cPlotBayesactThread()
        plotPanel = plotter.initFrame()
        plotter.initPlotBayesactSim(plotPanel)
        oBayesactSim.plotter = plotter

        bayesactSimThread = threading.Thread(target=oBayesactSim.startBayesactSim)
        plotter.setThread(bayesactSimThread)
        plotter.startApp()


if __name__ == "__main__":
    main(sys.argv)

