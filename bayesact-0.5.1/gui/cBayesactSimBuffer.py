# Buffer mechanism, it is possible to load this up earlier if you want
# But if bayesact is slow, it is possible that the plotting will catch up given enough speed
import subprocess
import sys
import os
import re
import signal
import ctypes
from cEnum import eAxes, eTurn, eEPA
from cConstants import cParseConstants, cSystemConstants, cWindowsOSConstants, cEPAConstants


class cBayesactSimBuffer:
    def __init__(self):
        self.m_Process = None
        self.m_KeepAlive = True

        self.m_MeanLearnerFlag = False
        self.m_MeanSimulatorFlag = False

        # Each element in the samples buffer will represent one agent client pair
        # and all its samples for ONE iteration phase
        # Each element contains [simulatorSamples, learnerSamples]
        self.m_SamplesBuffer = []

        # The buffer threshold will tell the program to try to sleep the bayesactsim
        # when there the number of elements in the samples buffer exceed the threshold
        # MAGIC NUMBER
        self.m_BufferThreshold = 1000

        # How it is stored:
        # [ [] [] [] [] [] [] [] [] [] ]
        # Where each [] is the total accumulation of a specific fundamental from all the samples
        # For example, the first [] will have all the self evaluations from all the samples
        # This is how the data will be stored and used for plotting
        self.m_SimulatorSamples = [[],[],[],
                                   [], [], [],
                                   [], [], []]
        self.m_LearnerSamples = [[],[],[],
                                   [], [], [],
                                   [], [], []]

        # learner is 0
        # simulator is 1
        self.m_Turn = -1


    def parseStream(self, iStream):
        for line in iStream:
            # To make output
            #self.m_Output.write(line)

            if (cParseConstants.m_SamplesPhrase in line):
                self.m_Turn = (self.m_Turn + 1) % 2
                self.m_MeanLearnerFlag = False

            if (cParseConstants.m_IterationPhrase in line):
                # Will sleep process when samples buffer exceed threshold
                if (len(self.m_SamplesBuffer) < self.m_BufferThreshold):
                    self.continueProcess()
                else:
                    self.sleepProcess()
                    # To make output
                    #self.m_KeepAlive = False


                self.m_SamplesBuffer.append([self.m_SimulatorSamples, self.m_LearnerSamples])

                self.m_SimulatorSamples = [[],[],[],
                                           [], [], [],
                                           [], [], []]
                self.m_LearnerSamples = [[],[],[],
                                           [], [], [],
                                           [], [], []]


                # To make output
                #print len(self.m_SamplesBuffer)

            if ((cParseConstants.m_MeanLearnerPhrase1 in line) or
                (cParseConstants.m_MeanLearnerPhrase2 in line) or
                (cParseConstants.m_MeanSimulatorPhrase1 in line) or
                (cParseConstants.m_MeanSimulatorPhrase2 in line)):
                self.m_MeanLearnerFlag = True

            if (True == self.m_MeanLearnerFlag):
                continue

            # Make GUI simple, don't give user too many choices to choose from
            # Get first 3 and last 3 for each agent and client
            if (None != re.search(cParseConstants.m_fValuesPhrase, line)):
                line = line.rstrip()
                # f values extract operation
                # there are only 9 attributes
                # The number 5 comes from when the array starts
                attributes = map(lambda x: float(x), filter(lambda y: "" != y, line[5:-1].split(" ")))

                if (eTurn.learner == self.m_Turn):
                    for i in range(cEPAConstants.m_NumAttributes):
                        self.m_SimulatorSamples[i].append(attributes[i])

                # else turn is simulator
                else:
                    for i in range(cEPAConstants.m_NumAttributes):
                        self.m_LearnerSamples[i].append(attributes[i])


    def run(self, iFileName=None):
        stream = None
        #print iFileName
        if (None == iFileName):
            self.m_Process = subprocess.Popen(cParseConstants.m_Command, shell=True, stdout=subprocess.PIPE)
            stream = self.m_Process.stdout
        else:
            stream = open(iFileName)

        # To make output
        #self.m_Process = subprocess.Popen(cParseConstants.m_Command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        #for line in iter(self.m_Process.stdout.readline, ""):
        #    sys.stdout.write('\r' + line[:-1])
        #    sys.stdout.flush()

        while(self.m_KeepAlive):
            self.parseStream(stream)
            if (None != iFileName):
                self.m_KeepAlive = False

        if (self.isMacOrLinux()):
            os.kill(self.m_Process.pid, signal.SIGKILL)

        else:
            print "exiting"
            # For some reason kernel32.OpenProcess is undocumented
            #handle = ctypes.windll.kernel32.OpenProcess(cWindowsOSConstants.m_ProcessTerminate, False, self.m_Process.pid)
            #ctypes.windll.kernel32.TerminateProcess(handle, -1)
            #ctypes.windll.kernel32.CloseHandle(handle)
            pass


    # Parses file with the data
    def parseFile(self, iFileName):
        with open(iFileName) as stream:
            lines = readlines(stream)
            self.parseProcess(lines)


    # Currently, I can only sleep and continue these processes on a Linux or Mac OS
    # Sleeping a processes on a Windows machine has not been implemented yet
    # Sleeping is not entirely necessary, just that the bayesact will constantly output numbers
    def sleepProcess(self):
        if (None == self.m_Process):
            return
        if (self.isMacOrLinux()):
            os.kill(self.m_Process.pid, signal.SIGSTOP)
        # else is WindowsOS
        else:
            # May need to implement a process sleeper for windows in the future
            pass


    def continueProcess(self):
        if (None == self.m_Process):
            return
        elif (self.isMacOrLinux()):
            os.kill(self.m_Process.pid, signal.SIGCONT)
        else:
            pass


    def getSamples(self):
        if (0 < len(self.m_SamplesBuffer)):
            return self.m_SamplesBuffer.pop(0)
        else:
            return None

    def isMacOrLinux(self):
        if (cSystemConstants.m_OS == cSystemConstants.m_LinuxOS or cSystemConstants.m_OS == cSystemConstants.m_MacOS):
            return True
        elif (cSystemConstants.m_OS == cSystemConstants.m_WindowsOS):
            return False
