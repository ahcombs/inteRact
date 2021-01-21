# Plots bayesactsim by parsing it
import subprocess
import os
import re
import time
import signal
import threading
import wx
from cBayesactSimBuffer import cBayesactSimBuffer
from cEnum import eAxes, eTurn, eEPA
from cConstants import cPlotConstants, cEPAConstants

class cPlotBayesactSim(object):
    # The axis items are enumerations of the EPA
    def __init__(self, iPlotEPAPanel):
        self.m_PlotEPAPanels = [iPlotEPAPanel]

        self.m_KeepAlive = True
        self.m_Lock = threading.Lock()

        self.m_LearnerFundamentalSamples = []
        self.m_SimulatorFundamentalSamples = []

        self.m_LearnerTauSamples = []
        self.m_SimulatorTauSamples = []

        self.m_LearnerPreviousAction = []
        self.m_SimulatorPreviousAction = []
        
        self.m_Sleep = False

        self.m_Parser = None
        self.m_ParserThread = None


    def plotBayesactSim(self):
        if ((None == self.m_LearnerFundamentalSamples) and (None == self.m_LearnerTauSamples)):
            return

        # To plot only a certain number of samples
        # Remember that this is a transpose of the samples, so the length of the learn/simul samples array is 9, the 3 pairs of epa
        for i in range(len(self.m_LearnerFundamentalSamples)):
            self.m_LearnerFundamentalSamples[i] = self.m_LearnerFundamentalSamples[i][:cPlotConstants.m_MaxPlotSamples]

        for i in range(len(self.m_SimulatorFundamentalSamples)):
            self.m_SimulatorFundamentalSamples[i] = self.m_SimulatorFundamentalSamples[i][:cPlotConstants.m_MaxPlotSamples]

        for i in range(len(self.m_LearnerTauSamples)):
            self.m_LearnerTauSamples[i] = self.m_LearnerTauSamples[i][:cPlotConstants.m_MaxPlotSamples]

        for i in range(len(self.m_SimulatorTauSamples)):
            self.m_SimulatorTauSamples[i] = self.m_SimulatorTauSamples[i][:cPlotConstants.m_MaxPlotSamples]


        for plotPanel in self.m_PlotEPAPanels:
            #with plotPanel.m_Lock
            if (eEPA.fundamental == plotPanel.m_PlotType):
                plotPanel.plotEPA(self.m_LearnerFundamentalSamples, self.m_SimulatorFundamentalSamples, self.m_LearnerPreviousAction, self.m_SimulatorPreviousAction)
            else:
                plotPanel.plotEPA(self.m_LearnerTauSamples, self.m_SimulatorTauSamples, self.m_LearnerPreviousAction, self.m_SimulatorPreviousAction)


    # Resplots and assumes the samples were already truncated to the max plot samples from the above function
    def replotOnPanel(self, iPlotEPAPanel):
        if (eEPA.fundamental == iPlotEPAPanel.m_PlotType):
            iPlotEPAPanel.plotEPA(self.m_LearnerFundamentalSamples, self.m_SimulatorFundamentalSamples, self.m_LearnerPreviousAction, self.m_SimulatorPreviousAction)
        else:
            iPlotEPAPanel.plotEPA(self.m_LearnerTauSamples, self.m_SimulatorTauSamples, self.m_LearnerPreviousAction, self.m_SimulatorPreviousAction)


    def sleepProcess(self):
        self.m_Sleep = True
        self.m_Parser.sleepProcess()


    def continueProcess(self):
        self.m_Sleep = False
        self.m_Parser.continueProcess()


    def killProcess(self):
        self.sleepProcess()
        self.m_Parser.m_KeepAlive = False



    def bufferData(self):
        self.m_Parser = cBayesactSimBuffer()
        self.m_ParserThread = threading.Thread(target=self.m_Parser.run)
        self.m_ParserThread.daemon=True
        self.m_ParserThread.start()
        while(self.m_Parser.m_BufferThreshold < len(self.m_Parser.m_SamplesBuffer)):
            print self.m_Parser.m_BufferThreshold


    def plotBufferedData(self):
        while(0 < len(self.m_Parser.m_SamplesBuffer)):
            self.plotBayesactSim()
            print self.m_Parser.m_BufferThreshold


    def plotFile(self, iFileName):
        self.m_Parser = cBayesactSimBuffer()
        self.m_Parser.parseFile(iFileName)
        self.plotBufferedData()


    def clearPlots(self):
        for panel in self.m_PlotEPAPanels:
            panel.clearAxes()
            panel.redrawAxes()


    def runOnPlot(self):
        # It is possible that you may preload data for the plot in the buffer
        # and then assign this plotter to a plot
        # This statement here prevents it though
        if (None == self.m_PlotEPAPanels[0]):
            # Thread ends
            return

        self.m_Parser = cBayesactSimBuffer()
        self.m_ParserThread = threading.Thread(target=self.m_Parser.run, kwargs={"iFileName" : None})
        self.m_ParserThread.daemon=True
        self.m_ParserThread.start()
        while(self.m_KeepAlive):
            while (not(self.m_Sleep)):
                fundamantals = self.m_Parser.getSamples()
                self.setSamples(fundamantals[eTurn.learner], fundamantals[eTurn.simulator])
                self.plotBayesactSim()
        self.killProcess()
        #self.m_ParserThread.join()


    def setFundamentals(self, iLearnerFundamentalSamples, iSimulatorFundamentalSamples):
        self.m_LearnerFundamentalSamples = iLearnerFundamentalSamples
        self.m_SimulatorFundamentalSamples = iSimulatorFundamentalSamples

    def setTau(self, iLearnerTauSamples, iSimulatorTauSamples):
        self.m_LearnerTauSamples = iLearnerTauSamples
        self.m_SimulatorTauSamples = iSimulatorTauSamples
