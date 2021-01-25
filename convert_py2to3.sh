#convert all bayesact base code to python 3 from python 2
#this enables it to be used by reticulate
#write new files back to the original names

#2to3 -w -n bayesact-0.5.1/bayesact.py
#2to3 -w -n bayesact-0.5.1/bayesactemot.py
#2to3 -w -n bayesact-0.5.1/bayesactsim.py
#2to3 -w -n bayesact-0.5.1/bayesactinteractive.py
#2to3 -w -n bayesact-0.5.1/pomcp.py

autopep8 --in-place --select E101,E11,E121,E122,E125,E126,E127,E128,E129,E131,E133,W690 bayesact-0.5.1/bayesact.py
autopep8 --in-place --select E101,E11,E121,E122,E125,E126,E127,E128,E129,E131,E133,W690 bayesact-0.5.1/bayesactemot.py
autopep8 --in-place --select E101,E11,E121,E122,E125,E126,E127,E128,E129,E131,E133,W690 bayesact-0.5.1/bayesactsim.py
autopep8 --in-place --select E101,E11,E121,E122,E125,E126,E127,E128,E129,E131,E133,W690 bayesact-0.5.1/bayesactinteractive.py
autopep8 --in-place --select E101,E11,E121,E122,E125,E126,E127,E128,E129,E131,E133,W690 bayesact-0.5.1/pomcp.py
