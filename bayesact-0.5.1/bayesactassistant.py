"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Assistance Interactive Example
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

#new verison is bayesactp - use just 'bayesact' to revert to the previous version
#also change in subclasses e.g. assistant.py and pwid.py
from bayesact import *
from assistant import *
from pwid import *

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
agent_knowledge=2

client_knowledge=2

# agent gender
agent_gender="male"

# client gender
client_gender="male"

#possibly set the agent id to be something
agent_id="assistant"
#if not in database (including "") then it is a randomly drawn id
#agent_id=""

#can also set the client id here if agent_knowledge = 2 (knows id of client - see above)
#if agent_knowledge is 0 then this is ignored
client_id = "elder"

#client_id = [-0.76,-0.2,-0.88]  #bitter elder
#client_id = [-0.66,-0.43,-1.80] #lonesome elder
#client_id = "boss"

#what is the client really? 
#true_client_id = "patient"
true_client_id = "elder"   # 1.59, 0.79, -0.88
#true_client_id = [-0.76,-0.2,-0.88]  #bitter elder - transient impression is -0.76, -0.20, -0.88
#true_client_id = [-0.66,-0.43,-1.80] #lonesome elder
#true_client_id = "psychotic"
#client_id = ""

#initial awareness distribution 0 = aware, 1 = unaware
initial_px = [0.3,0.7]

#set to "" to not do a fixed action
fixed_agent_act=""

#when there is a fixed agent action, this is the action taken when no prompt
fixed_agent_default_act="mind"

if len(sys.argv) > 1:
    true_client_id=sys.argv[1]
if len(sys.argv) > 2:
    fixed_agent_act=sys.argv[2]
if len(sys.argv) > 3:
    fixed_agent_default_act=sys.argv[3]


print "true_client_id: ",true_client_id
print "fixed_agent_act: ",fixed_agent_act
print "fixed_agent_default_act: ",fixed_agent_default_act

#who goes first? 
initial_learn_turn="client"
initial_simul_turn="agent"



num_plansteps=8  #includes the zeroth plan step and the end planstep

num_behaviours=5

#how often do we want to see the full id sets learned by the agent
get_full_id_rate=-1


#do we want to try to mimic interact?
mimic_interact=False

#if True, don't ask client just run
do_automatic=False

use_pomcp=False

#parameters for POMCP
numcact=5  #user defined 
numdact=(num_plansteps-1)+1  #8 planstep changes  +  1 null action
obsres=1.0
actres=1.0
timeout=10.0

if not fixed_agent_act=="":
    numdact=1
    numcact=1   ##?? why 1 here - should stay the same at 9?

print "do_pomcp? :",use_pomcp
print "pomcp obsres:  ",obsres
print "pomcp actres:  ",actres
print "pomcp timeout: ",timeout
print "pomcp numcact: ",numcact
print "pomcp numdact: ",numdact

max_num_iterations=50


#dictionary giving the state dynamics 
nextPsDict={0:{0:([1.0],[0]),1:([1.0],[2]),2:([0.8,0.2],[1,0]),3:([1.0],[0]),4:([1.0],[0])},
            1:{0:([1.0],[1]),1:([1.0],[3]),2:([0.8,0.2],[1,0]),3:([1.0],[1]),4:([1.0],[1])},
            2:{0:([1.0],[2]),1:([1.0],[2]),2:([0.8,0.2],[3,2]),3:([1.0],[2]),4:([1.0],[0])},
            3:{0:([1.0],[3]),1:([1.0],[3]),2:([0.6,0.4],[3,2]),3:([1.0],[4]),4:([1.0],[1])},
            4:{0:([1.0],[4]),1:([1.0],[3]),2:([0.8,0.2],[6,4]),3:([1.0],[4]),4:([1.0],[5])},
            5:{0:([1.0],[5]),1:([1.0],[3]),2:([0.8,0.2],[7,5]),3:([1.0],[4]),4:([1.0],[5])},
            6:{0:([1.0],[6]),1:([1.0],[2]),2:([0.8,0.2],[6,4]),3:([1.0],[6]),4:([1.0],[7])},
            7:{0:([1.0],[7]),1:([1.0],[2]),2:([0.6,0.4],[7,5]),3:([1.0],[7]),4:([1.0],[7])}}



nextBehDict={0:([0.8,0.2],[2,1]),
             1:([1.0],[1]),
             2:([1.0],[2]),
             3:([1.0],[3]),
             4:([0.6,0.4],[2,4]),
             5:([1.0],[2]),
             6:([1.0],[4]),
             7:([1.0],[0])}



nextPsDictlinear={0:([1.0],[1]),
            1:([1.0],[2]),
            2:([1.0],[3]),
            3:([1.0],[4]),
            4:([1.0],[5]),
            5:([1.0],[6]),
            6:([1.0],[7]),
            7:([1.0],[7])}



#-----------------------------------------------------------------------------------------------------------------------------
#these parameters can be tuned, but using caution
#-----------------------------------------------------------------------------------------------------------------------------
#agent's ability to change identities - higher means it will shape-shift more
bvagent=0.0001
#agent's belief about the client's ability to change identities - higher means it will shape-shift more
bvclient=0.0001  #was 0.5

simul_bvclient=0.001
simul_bvagent=0.001

#initial estimates of ids
#learn_beta_agent_init is given by init_id, the others are overridden here
#agent_learn_beta_client_init=2.0     #0.0001

#pwd has a vague identity and a rough concept of the identity of the agent
simul_learn_beta_agent_init=0.0001
simul_learn_beta_client_init=0.0001

#-----------------------------------------------------------------------------------------------------------------------------
#these parameters can be tuned, but will generally work "out of the box" for a basic simulation
#-----------------------------------------------------------------------------------------------------------------------------

#behaviours file
fbfname="fbehaviours.dat"

#identities file
fifname="fidentities.dat"

#get some key parameters from the command line
#set much larger to mimic interact (>5000)
num_samples=2000


# use 0.0 for mimic interact simulations
#roughening_noise=0.0
roughening_noise=num_samples**(-1.0/3.0)


#the observation noise
#set to 0.05 or less to mimic interact
#in our current assistant, the EPA measurements are null on the E dimension (we can't measure this yet)
# and pretty bad on the other two, so we adjust here for that
#obs_noise=(100000,3.5,1.5)
obs_noise=(100000,0.5,0.5)
#obs_noise=(100000,80,50)
#obs_noise=(0.5,0.5,0.5)
#obs_noise=0.5

simul_obs_noise=0.5

xobsnoise=0.1
simul_xobsnoise=0.00001

if mimic_interact:
    #bvagent_init=0.000001
    bvclient_init=0.000001
    #bvagent=0.00001
    bvclient=0.00001
    agent_knowledge=2
    num_samples=100000
    roughening_noise=0.0
    obs_noise=0.01
    num_action_samples=10000


#do we print out all the samples each time
learn_verbose=False

#for repeatability
rseed = NP.random.randint(0,382948932)
#rseed=259721666
rseed=3620463
print "random seeed is : ",rseed
NP.random.seed(rseed)

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

        
    
#a function to ask the client for an action to take and map it to an EPA value
def ask_client(fbeh,sugg_act='',epaact=[],qq="behaviour"):
    sst="your action ("+qq+" ... '?' shows options): "

    sstnc="You can now either :\n"
    sstnc+="- pick an action ("+qq+") and type in its label\n"
    sstnc+="- type 'quit' to stop the simulation\n"
    sstnc+="- type a comma separated list of three numbers to specify the action as E,P,A values\n"
    if not sugg_act=='':
        sstnc+="- hit 'enter' to take default action with label: "+sugg_act+"\n"
    if not epaact=='':
        sstnc+="- type any digit (0-9) to take suggsted action: "+str(epaact)

    if do_automatic:
        return epaact

    
    while True:
        cact = raw_input(sst)
        if cact=='quit':
            return []
        elif cact=='' and not sugg_act=='':
            cact=sugg_act
            if qq=="identity":
                return cact
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
    print "client said: ",cact
    return observ

fbehaviours_agent=readSentiments(fbfname,agent_gender)
fbehaviours_client=readSentiments(fbfname,client_gender)

(agent_mean_ids,agent_cov_ids)=getIdentityStats(fifname,agent_gender)
(client_mean_ids,client_cov_ids)=getIdentityStats(fifname,client_gender)

#the mean and covariance of IDs for male agents as taken from the databases
#should do this automatically in python based on actual genders of client/agent....as above now
#mean_ids=NP.array([0.40760,0.40548,0.45564])
#$cov_ids=NP.array([[2.10735,1.01121, 0.48442],[1.01121,1.22836,0.55593],[0.48442,0.55593,0.77040]])

#the actual (true) ids drawn from the distribution over ids, if not set to something in particular
aid=agent_id
agent_id=getIdentity(fifname,agent_id,agent_gender)
if agent_id==[]:
    agent_id=NP.random.multivariate_normal(agent_mean_ids,agent_cov_ids)
agent_id=NP.asarray([agent_id]).transpose()

if type( client_id ) == type( "random string" ):
    client_id=getIdentity(fifname,client_id,agent_gender)
if client_id==[]:
    client_id =  NP.random.multivariate_normal(client_mean_ids,client_cov_ids)
client_id=NP.asarray([client_id]).transpose()


client_agent_id=getIdentity(fifname,aid,client_gender)
if client_agent_id==[]:
    client_agent_id=NP.random.multivariate_normal(agent_mean_ids,agent_cov_ids)
client_agent_id=NP.asarray([client_agent_id]).transpose()



if type( true_client_id ) == type( "random string" ):
    #don't need to do this translation if the true_client_id is an EPA value
    true_client_id=getIdentity(fifname,true_client_id,client_gender)
tcid=true_client_id
if true_client_id==[]:
    true_client_id =  NP.random.multivariate_normal(client_mean_ids,client_cov_ids)
true_client_id=NP.asarray([true_client_id]).transpose()

#get initial sets of parameters for agent
#(learn_tau_init,learn_prop_init,learn_beta_client_init,learn_beta_agent_init)=init_id(agent_knowledge,agent_id,client_id,client_mean_ids)
(learn_tau_init,learn_prop_init,learn_beta_client_init,learn_beta_agent_init)=init_id(agent_knowledge,agent_id,client_id,tcid)

learn_beta_client_init = 2.0  #override


(simul_tau_init,simul_prop_init,simul_beta_client_init,simul_beta_agent_init)=init_id(client_knowledge,true_client_id,client_agent_id,agent_mean_ids)

#overwrite these values for the interactive script only
#do this for mimicking interact
if mimic_interact:
    learn_beta_agent_init=bvagent_init
    learn_beta_client_init=bvclient_init


if not fixed_agent_act=="":
    fbehv=fbehaviours_agent[fixed_agent_act]
    fixed_agent_aab=[float(fbehv["e"]),float(fbehv["p"]),float(fbehv["a"])]
    print "agent taking fixed action: ",fixed_agent_act," which is ",fixed_agent_aab

    fbehv=fbehaviours_agent[fixed_agent_default_act]
    fixed_agent_default_aab=[float(fbehv["e"]),float(fbehv["p"]),float(fbehv["a"])]
    print "agent taking fixed default action: ",fixed_agent_default_act," which is ",fixed_agent_default_aab
else:
    print "agent taking bayesact actions"

#set up fixed policy
#this is an array fixed_policy[i] giving the affective action to take for 
#each discrete action available
fixed_policy=None
if not fixed_agent_act=="":
    fixed_policy={}
    fixed_policy[0]=mapSentimentLabeltoEPA(fbehaviours_agent,fixed_agent_default_act)

    for a in range(numdact):
        fixed_policy[a+1]=mapSentimentLabeltoEPA(fbehaviours_agent,fixed_agent_act)

print "fixed policy is: "
print fixed_policy



#get the agent - can use some other subclass here if wanted 
learn_agent=Assistant(N=num_samples,alpha_value=1.0,
                      gamma_value=obs_noise,beta_value_agent=bvagent,beta_value_client=bvclient,
                      #beta_value_client_init=agent_learn_beta_client_init,    #this override was confusing things - why was this here?
                      beta_value_client_init=learn_beta_client_init,
                      beta_value_agent_init=learn_beta_agent_init,
                      client_gender=client_gender,agent_gender=agent_gender,use_pomcp=use_pomcp,
                      agent_rough=roughening_noise,client_rough=roughening_noise, nextpsd = nextPsDict, nextbd = nextBehDict, onoise=xobsnoise, 
                      fixed_policy=fixed_policy,pomcp_interactive=True,
                      numcact=numcact,numdact=numdact,obsres=obsres,actres=actres,pomcp_timeout=timeout)


#get the agent - can use some other subclass here if wanted 
simul_agent=PwD(N=num_samples,alpha_value=1.0,
                gamma_value=simul_obs_noise,beta_value_agent=simul_bvagent,beta_value_client=simul_bvclient,
                beta_value_client_init=simul_learn_beta_client_init,beta_value_agent_init=simul_learn_beta_agent_init,
                client_gender=agent_gender,agent_gender=client_gender,
                agent_rough=roughening_noise,client_rough=roughening_noise, nextpsd = nextPsDict, nextbd = nextBehDict, onoise=simul_xobsnoise,
                numcact=numcact,numdact=numdact,obsres=obsres,actres=actres,pomcp_timeout=timeout)

learn_initx=[initial_learn_turn,initial_px]
simul_initx=[initial_simul_turn,initial_px]



print 10*"-","learning agent parameters: "
learn_agent.print_params()
print "learner init tau: ",learn_tau_init
print "learner prop init: ",learn_prop_init
print "learner beta client init: ",learn_beta_client_init
print "learner beta agent init: ",learn_beta_agent_init

print 10*"-","simulated agent parameters: "
simul_agent.print_params()

learn_avgs=learn_agent.initialise_array(learn_tau_init,learn_prop_init,learn_initx)

print "learner (assistant) average sentiments (f) after initialisation: "
learn_avgs.print_val()

simul_avgs=simul_agent.initialise_array(simul_tau_init,simul_prop_init,simul_initx)
print "simulator (pwid) average sentiments (f) after initialisation: "
simul_avgs.print_val()


#cval,numaact,numpact,actres,obsres):
#learn_agent.POMCP_initialise(c_val,numact,num_plansteps+1,actres,obsres,timeout,fixed_policy)
#learn_agent.POMCP_initialise(c_val,numact,num_plansteps+1,obsres,timeout,fixed_policy)
#if use_pomcp:
#    learn_agent.initialise_pomcp(cval,

learn_turn=initial_learn_turn
simul_turn=initial_simul_turn    



done = False
iter=0
ps_obs=0
while not done:
    print 10*"#"," current turn: ",learn_turn," ",10*"#"

    observ=[]
    print 10*"-","iter ",iter,80*"-"




    (learn_aab,learn_paab)=learn_agent.get_next_action(learn_avgs)
    print "agent action/client observ: ",learn_aab        
    simul_observ=learn_aab
    print "agent prop. action: ",learn_paab

    if not fixed_agent_act=="":
        if learn_paab==None:
            learn_aab=fixed_agent_default_aab
        else:
            learn_aab=fixed_agent_aab
            #learn_aab=[0.15,0.32,0.06]   #force prompt every time
            #learn_aab=[0.43,0.61,0.5]  #force cue every time
            #learn_aab=[-0.09, 1.29, 1.59]  #force command every time

    




    if learn_turn=="agent":
        learn_aact=findNearestBehaviour(learn_aab,fbehaviours_agent)
        print "suggested action for the agent is :",learn_aab,"\n  closest label is: ",learn_aact
        print "agent's proposition action : ",learn_paab,"\n"
        

        print "enter an action for the agent to take, or '0' for the Bayesact choice"
        learn_aab=ask_client(fbehaviours_agent,learn_aact,learn_aab)
        print "agent does action :",learn_aab,"\n"
        simul_observ=learn_aab
        learn_observ=[]  #awkward

        #remember for next update of pwid
        simul_agent.set_last_prompt(learn_paab)

    (simul_aab,simul_paab)=simul_agent.get_next_action(simul_avgs)
    print "client action/agent observ: ",simul_aab
    learn_observ=simul_aab
    print "client prop. action: ",simul_paab


    if simul_turn=="agent":
        simul_aact=findNearestBehaviours(simul_aab,fbehaviours_agent,10)
        print "client's proposition action : ",simul_paab,"\n"
        print "agent advises the following action :",simul_aab,"\n  closest labels are: ", [re.sub(r"_"," ",i.strip()) for i in simul_aact]
        
        
        #now, this is where the client actually decides what to do, possibly looking at the suggested labels 
        simul_aab=ask_client(fbehaviours_client,simul_aact[0],simul_aab)
        print "client does action: ",simul_aab,"\n"
        learn_observ=simul_aab
        simul_observ=[]  #awkward

    

    
    #observation of behaviour
    if do_automatic:
        #really, this would be  - get observation from handtracker and convert to planstep observation
        #here, we use this simple hack to do this automatically, but this won't work in all cases
        beh_obs=simul_paab
    elif simul_turn=="agent":
        gotbeh=False
        while not gotbeh:
            beh_obs = raw_input("Enter behaviour observation (from 0 to "+str(num_behaviours-1)+") : ")
            try:
                beh_obs = int(beh_obs)
                if beh_obs>=0 and beh_obs<num_behaviours:
                    gotbeh=True
            except ValueError:
                gotbeh=False
        simul_agent.set_behaviour(beh_obs)
    else:
        beh_obs=simul_paab

    print "agent planstep distribution before update: "
    curr_planstep = learn_agent.get_most_likely_planstep()


    if learn_turn=="client" and learn_observ==[]:
        done = True
    elif learn_turn=="agent" and learn_aab==[]:
        done = True
    elif iter > max_num_iterations:
        done = True
    elif simul_agent.is_done():
        print "all done"
        done = True
    else:
        learn_xobs=[State.turnnames.index(invert_turn(learn_turn)),beh_obs]
        learn_avgs=learn_agent.propagate_forward(learn_aab,learn_observ,xobserv=learn_xobs,paab=learn_paab,verb=learn_verbose)

        print "agent f is: "
        learn_avgs.print_val()
        
        print "agent planstep distribution: "
        curr_planstep = learn_agent.get_most_likely_planstep()



        #learn_paab is passed into client as the x-observation
        simul_xobs=[State.turnnames.index(invert_turn(simul_turn)),learn_paab]
        simul_avgs=simul_agent.propagate_forward(simul_aab,simul_observ,xobserv=simul_xobs,paab=None,verb=learn_verbose)

        print "client f is: "
        simul_avgs.print_val()


        #I think these should be based on fundamentals, not transients
        (aid,cid)=learn_agent.get_avg_ids(learn_avgs.f)
        print "agent thinks it is most likely a: ",aid
        print "agent thinks the client is most likely a: ",cid

        (aid,cid)=simul_agent.get_avg_ids(simul_avgs.f)
        print "client thinks it is most likely a: ",aid
        print "client thinks the agent is most likely a: ",cid
        
        print "client state is: "
        simul_agent.print_state()

        if get_full_id_rate>0 and (iter+1)%get_full_id_rate==0:
            (cnt_ags,cnt_cls)=learn_agent.get_all_ids()
            print "agent thinks of itself as (full distribution): "
            print cnt_ags[0:10]
            print "agent thinks of the client as (full distribution): "
            print cnt_cls[0:10]
        iter += 1
    print "current deflection of averages: ",learn_agent.deflection_avg

    print "current deflection of averages (client): ",simul_agent.deflection_avg
            
    learn_d=learn_agent.compute_deflection()
    print "current deflection (agent's perspective): ",learn_d

    simul_d=simul_agent.compute_deflection()
    print "current deflection (client's perspective): ",simul_d
    if learn_turn=="client":
        learn_turn="agent"
    elif learn_turn=="agent":
        learn_turn="client"

    if simul_turn=="client":
        simul_turn="agent"
    elif simul_turn=="agent":
        simul_turn="client"
    

print "final simul agent state: "
simul_agent.print_state()
