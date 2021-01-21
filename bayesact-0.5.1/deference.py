"""------------------------------------------------------------------------------------------
Bayesian Affect Control Theory
Deference Code - to run the experiments discussed in the paper 
Rob Freeland and Jesse Hoey. The Structure of Deference: Modeling Occupational Status Using Affect Control Theory. To appear. 

Author (of this code): Jesse Hoey  jhoey@cs.uwaterloo.ca   http://www.cs.uwaterloo.ca/~jhoey
December 2017
Use for research purposes only.
Please do not re-distribute without written permission from the author
Any commerical uses strictly forbidden.
Code is provided without any guarantees.
Research sponsored by the Natural Sciences and Engineering Council of Canada (NSERC).
use python 2
----------------------------------------------------------------------------------------------"""
from bayesact import *
import getopt
import sys

verbose=False



#reads in SD vals for labels 
def readSDvals(fname):
    #open the input file for fundamentals of behaviours or identities
    fmeanvals={}
    fsdvals={}
    fb=file(fname,"rU")
    for x in set(fb):
        y=[re.sub(r"\s+","_",i.strip()) for i in x.split(",")]
        fsdvals[y[0]]=map(lambda x: float(x), y[4:7])
        fmeanvals[y[0]]=map(lambda x: float(x), y[1:4])
    fb.close()
    return [fmeanvals,fsdvals]

def readMEANvals(fname):
    #open the input file for fundamentals of behaviours or identities
    fmeanvals={}
    fb=file(fname,"rU")
    for x in set(fb):
        y=[re.sub(r"\s+","_",i.strip()) for i in x.split(",")]
        fmeanvals[y[0]]=map(lambda x: float(x), y[1:4])
    fb.close()
    return fmeanvals

#reads in COV vals for labels
def readCOVvals(fname):
    #open the input file for fundamentals of behaviours or identities
    fcovvals={}
    fmeanvals={}
    fb=file(fname,"rU")
    for x in set(fb):
        y=[re.sub(r"\s+","_",i.strip()) for i in x.split(",")]
        fcovvals[y[0]]=map(lambda x: float(x), y[4:13])
        fmeanvals[y[0]]=map(lambda x: float(x), y[1:4])
    fb.close()
    return [fmeanvals,fcovvals]

def write_out_deflection(agent):
    for sample in agent.samples:
        deflection = sample.weight*NP.dot(sample.f-sample.tau,sample.f-sample.tau)
        print ' '.join('{:3.2f}'.format(k) for k in sample.f),
        print ' '.join('{:3.2f}'.format(k) for k in sample.tau),
        print '{:3.2f}'.format(deflection)

def main(argv):
    #-----------------------------------------------------------------------------------------------------------------------------
    #user-defined parameters
    #-----------------------------------------------------------------------------------------------------------------------------
    #agent knowledge of client id - this must be 2
    agent_knowledge=2

    #who goes first?
    initial_turn="agent"

    #use  the standard deviations in identities for the initial identity distributions
    use_id_SD=False
    
    #if true, read full covariances in 
    use_full_COV=False

    #will use this action:
    # set to defer_to for final version
    deference_action="defer_to"



    #-----------------------------------------------------------------------------------------------------------------------------
    #these parameters can be tuned, but using caution
    #-----------------------------------------------------------------------------------------------------------------------------
    #agent's ability to change identities - higher means it will shape-shift more
    bvagent=0.0001
    #agent's belief about the client's ability to change identities - higher means it will shape-shift more
    bvclient=0.0001

        
    #-----------------------------------------------------------------------------------------------------------------------------
    #these parameters can be tuned, but will generally work "out of the box" for a basic simulation
    #-----------------------------------------------------------------------------------------------------------------------------
    #behaviours file 
    fbfname="fbehaviours.dat"

    ## this will be the default - to be overridden by the command line
    fifname="fidentities.dat"

    idtype="MEAN"
    #TODO:
    #read in file names for identities and behaviours on the command line
        
    #get some key parameters from the command line
    num_samples=500

    roughening_noise=num_samples**(-1.0/3.0)


    #do we print out all the samples each time
    learn_verbose=False

    output_fname="BayesACT"
    rseed = NP.random.randint(0,382948932)
    #for repeatability
    #rseed=271887164

    NP.random.seed(rseed)
    print "random seeed is : ",rseed

    
    agent_beh=deference_action
        
    helpstring="BayesACT Deference score calculator (2 agents) usage:\n python deference.py\n\t -n <number of samples (default 500)>\n\t -f outputfilename prefix (output files will start with this prefix. Default: BayesACT)\n\t -a deference behaviour (default: defer_to)\n\t -b behaviours filename (default: fbehaviours.dat)\n\t    behaviour files should be of type MEAN (see below)\n\t -i identities filename (default: fidentities.dat)\n\t -t type of identities (can be MEAN, COV or SD, default: MEAN)\n\t    identity files are comma separated (CSV) and have the following format:\n\t\t MEAN: each line has identity followed by 3 numbers for E,P,A.  In this case the standard deviation used for identities is 0.1\n\t\t SD: each line has identity string followed by 3 values for mean EPA followed by 3 values for Standard Deviation EPA\n\t\t COV: each line has identity string followed by 3 values for mean EPA followed by 9 values for the covariance matrix row-wise as EPA\n"
    #behaviours filename (defaults to fbehaviours.dat - will use "defer to" from it)
    #identity filename (defaults to fidentities.dat - will use SDs of beta_agent_init=0.1 as specified in init_id)
    #otherwise, identity file must have on each line:
    #identity mean_epa (3 values), cov_epa (9 values)
    #or
    #identity mean_epa (3 value), epa std. devs (3 values)
    

    try:
        opts, args = getopt.getopt(argv[1:],"hvn:f:i:b:t:a:",["help","verbose","n=","f=","i=","b=","t=","a="])
    except getopt.GetoptError:
        print helpstring
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print helpstring
            sys.exit()
        elif opt == "-v":
            learn_verbose=True
        elif opt in ("-n", "--numsamples"):
            num_samples = int(arg)
        elif opt in ("-f", "--outfname"):
            output_fname=arg
        elif opt in ("-i", "--identityfname"):
            fifname=arg
        elif opt in ("-b", "--behaviourfname"):
            fbfname=arg
        elif opt in ("-t", "--identitytype"):
            idtype=arg
        elif opt in ("-a", "--agentbeh"):
            agent_beh=arg

    if idtype=="COV":
        use_full_COV=True
    elif idtype=="SD":
        use_id_SD=True

        

    fbehaviours=readMEANvals(fbfname)
        
    if use_full_COV:
        [fidentities,fidentities_cov] = readCOVvals(fifname)
    elif use_id_SD:
        [fidentities,fidentities_sd] = readSDvals(fifname)
    else:
        fidentities = readMEANvals(fifname)
        
    deference_scores={}
    deference_avg_scores={}
    
    num_of_agents=0
    tot_num_agents=len(fidentities)
    start_time = time.time()

    # do all the identities specified
    agent_ids_to_do = fidentities
    client_ids_to_do = fidentities

        

    for agent_id in agent_ids_to_do:
        deference_scores[agent_id]={}
        deference_avg_scores[agent_id]={}
        for client_id in client_ids_to_do:
            #the actual (true) ids drawn from the distribution over ids, if not set to something in particular
            agent_id_vec=fidentities[agent_id]
            agent_id_vec=NP.asarray([agent_id_vec]).transpose()

            #here we get the identity of the client *as seen by the agent*
            ## OCT 17th 2019 - this used to read (wrongly):  - not used in ASR paper just in this new version of the code
            ###client_id_vec=fidentities[agent_id]
            
            client_id_vec=fidentities[client_id]
            client_id_vec=NP.asarray([client_id_vec]).transpose()

            #get initial sets of parameters for agent
            (learn_tau_init,learn_prop_init,learn_beta_client_init,learn_beta_agent_init)=init_id(agent_knowledge,agent_id_vec,client_id_vec)
        
            #initial x - only contains the turn for a default agent (no other x components)
            learn_initx=initial_turn

            
            
            learn_agent=Agent(N=num_samples,alpha_value=1.0,
                              gamma_value=0.1,beta_value_agent=bvagent,beta_value_client=bvclient,
                              beta_value_client_init=learn_beta_client_init,beta_value_agent_init=learn_beta_agent_init,
                              agent_rough=roughening_noise,client_rough=roughening_noise,
                              init_turn=initial_turn)
            learn_agent.verbose=verbose
            #### HERE we have to now change the agent's sigfi matrix so it contains the actual SD values
            #### changed this Sept 30th 2019 - was originally like this
            #if use_id_SD:
            #    if not use_full_COV:
            #           --- use sd[...]
            #    else:
            #           ---- use _cov[...]
            #which means it was using "mean" instead of "cov" - this was only put on the website though so I should correct that
            #revised version is :
        

            if use_id_SD:
                for i in [0,1,2]:
                    learn_agent.sigfi[i][i]=fidentities_sd[agent_id][i]**2
                for i in [0,1,2]:
                    learn_agent.sigfi[i+6][i+6]=fidentities_sd[client_id][i]**2
            elif use_full_COV:
                for i in [0,1,2]:
                    for j in [0,1,2]:
                        learn_agent.sigfi[i][j]=fidentities_cov[agent_id][i*3+j]
                for i in [0,1,2]:
                    for j in [0,1,2]:
                        learn_agent.sigfi[i+6][j+6]=fidentities_cov[client_id][i*3+j]

                        
            #min_eig = NP.min(NP.real(NP.linalg.eigvals(learn_agent.sigfi)))
            #if min_eig < 0:
            #    print "matrix not positive semi-definite! for identity agent:"+agent_id+" client: "+client_id
            #    print learn_agent.sigfi
                ## attempt to fix this - but I think I can ignore
                ##learn_agent.sigfi -= 10*min_eig * NP.eye(*(learn_agent.sigfi).shape)
            #print learn_agent.sigfi
            if verbose:
                print 10*"-","learning agent parameters: "
                learn_agent.print_params()
                print "learner init tau: ",learn_tau_init
                print "learner prop init: ",learn_prop_init
                print "learner beta client init: ",learn_beta_client_init
                print "learner beta agent init: ",learn_beta_agent_init
                
            #the following two initialisation calls should be inside the Agent constructor to keep things cleaner
            learn_avgs=learn_agent.initialise_array(learn_tau_init,learn_prop_init,learn_initx)
            #printRawSamples(learn_agent.samples)

            if verbose:
                print "learner average sentiments (f): "
                learn_avgs.print_val()



            learn_avgs = learn_agent.getAverageState()

            learn_turn=learn_avgs.get_turn()
            if verbose:
                print "agent state is: "
                learn_avgs.print_val()
                print learn_turn
                
            observ=[]
            (learn_aab,learn_paab)=learn_agent.get_next_action(learn_avgs,exploreTree=True)

            cact=agent_beh
            learn_aab=fbehaviours[cact]

            if verbose:
                print "agent does action :",learn_aab,"\n"

            learn_observ=[]
            learn_eobserv=[]
            
            #agent gets to observe the turn each time
            learn_xobserv=[State.turnnames.index(invert_turn(learn_turn))]
            
            #the main SMC update step
            learn_avgs=learn_agent.propagate_forward(learn_aab,learn_observ,learn_xobserv,learn_paab,verb=learn_verbose,agent=eTurn.learner)

            #I think these should be based on fundamentals, not transients
            learn_d=learn_agent.compute_deflection()
            if verbose:
                (aid,cid)=learn_agent.get_avg_ids(learn_avgs.f)
                print "agent thinks it is most likely a: ",aid
                print "agent thinks the client is most likely a: ",cid
                
                print "current deflection of averages: ",learn_agent.deflection_avg
                
                print "current deflection (agent's perspective): ",learn_d
                
                print 100*"-"
                print agent_id," ",agent_beh," ",client_id," ",learn_d
            deference_scores[agent_id][client_id]=learn_d
            #these are approximately the ACT scores - the deflections based on the average ids
            deference_avg_scores[agent_id][client_id]=learn_agent.deflection_avg
                
                
        num_of_agents += 1
        print "did id: ",agent_id," number of agents so far: ",num_of_agents,"/",tot_num_agents
        elapsed_time = time.time() - start_time
        print "time taken: ",elapsed_time,
        time_per_id=elapsed_time/num_of_agents
        print "time per id: ",time_per_id,
        time_remaining=(tot_num_agents-num_of_agents)*time_per_id/60.0
        print "estimated time remaining: ",time_remaining," minutes"," or ",time_remaining/60," hours"
        

    #print all out
    fid=open(output_fname+"DeflectionMatrix.txt","w")
    fid.write("identities:"+",")
    for client_id in client_ids_to_do:
        fid.write(client_id+",")
    fid.write("\n")
    for agent_id in agent_ids_to_do:
        fid.write(agent_id+",")
        for client_id in client_ids_to_do:
            fid.write(str(deference_scores[agent_id][client_id])+",")
        fid.write("\n")
    fid.close()



    #compute stuff
    mean_deference_score={}
    #actually negative log probability
    symmetric_deference_prob={}
    num_ids=0
    for agent_id in agent_ids_to_do:
        mean_deference_score[agent_id]=0
        symmetric_deference_prob[agent_id]=0
        num_ids += 1
        for client_id in client_ids_to_do:
            mean_deference_score[agent_id] += deference_scores[agent_id][client_id]
            symmetric_deference_prob[agent_id] += deference_scores[agent_id][client_id]
            symmetric_deference_prob[agent_id] += -1.0*math.log(1.0-math.exp(-1.0*deference_scores[client_id][agent_id]))
            
            
    for agent_id in agent_ids_to_do:
        mean_deference_score[agent_id]=mean_deference_score[agent_id]/num_ids
                
    #print mean_deference_score
    sorted_mds=sorted(mean_deference_score.items(), key=lambda x: x[1], reverse=True)
    fid=open(output_fname+"MeanDeflection.txt","w")
    for m in sorted_mds:
        fid.write(m[0]+","+str(m[1])+"\n")
    fid.close()
        
    fid=open(output_fname+"SymmetricDeflection.txt","w")
    sorted_mds=sorted(symmetric_deference_prob.items(), key=lambda x: x[1], reverse=True)
    for m in sorted_mds:
        fid.write(m[0]+","+str(m[1])+"\n")
    fid.close()
            
        

if __name__ == "__main__":
    main(sys.argv)
