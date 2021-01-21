import threading
#this was cPlotEPA3D - 
import cPlotEPA2D
from cPlotBayesactSim import cPlotBayesactSim
import wx
from cEnum import eEPA

class cPlotBayesactThread(object):
    def __init__(self):
        self.m_App = None
        self.m_PlotFrame = None

        self.m_PlotBayesactSim = None
        self.m_Thread = None
        self.m_ThreadExists = False

        self.m_LearnerFundamentalSamples = []
        self.m_SimulatorFundamentalSamples = []

        self.m_LearnerTauSamples = []
        self.m_SimulatorTauSamples = []

        self.m_LearnerPreviousAction = []
        self.m_SimulatorPreviousAction = []


    # Used when you want to just blatantly start a window and plot stuff on it, will probably be deprecated later
    def initFrame(self):
        self.m_App = wx.App(redirect=False)
        #this was cPlotEPA3D and then uncomment the #iZAxisItem line in the arguments of the call to initPanel
        self.m_PlotFrame = cPlotEPA2D.cPlotFrame(None, title="Bayesact Simulator", size=(700, 550))
        self.m_PlotFrame.initPanel(iXAxisItem=eEPA.evaluation,
                                   iYAxisItem=eEPA.potency,
                                   #iZAxisItem=eEPA.activity,
                                   style=wx.SIMPLE_BORDER, pos=(0, 0), size=(700, 550))
        return self.m_PlotFrame.m_PlotPanel


    # Used to bind the first plot panel
    def initPlotBayesactSim(self, iPlotPanel):
        self.m_PlotBayesactSim = cPlotBayesactSim(iPlotPanel)


    def addPanel(self, iPlotPanel):
        if (None == self.m_PlotBayesactSim):
            self.initPlotBayesactSim(iPlotPanel)
        else:
            self.m_PlotBayesactSim.m_PlotEPAPanels.append(iPlotPanel)


    def setThread(self, iThread):
        self.m_Thread = iThread
        iThread.daemon = True
        self.m_ThreadExists = True


    def startThread(self):
        self.m_Thread.start()


    # Used for standalone simulations, that is, with the init frame
    def startApp(self):
        if (False == self.m_ThreadExists):
            print "ERROR: Thread has not been declared"
        else:
            self.m_Thread.start()
            self.m_PlotFrame.Show()
            self.m_App.MainLoop()


    def plot(self):
        self.m_PlotBayesactSim.setFundamentals(self.m_LearnerFundamentalSamples, self.m_SimulatorFundamentalSamples)
        self.m_PlotBayesactSim.setTau(self.m_LearnerTauSamples, self.m_SimulatorTauSamples)
        self.m_PlotBayesactSim.m_LearnerPreviousAction = self.m_LearnerPreviousAction
        self.m_PlotBayesactSim.m_SimulatorPreviousAction = self.m_SimulatorPreviousAction
        self.m_PlotBayesactSim.plotBayesactSim()


    def replotOnPanel(self, iPlotEPAPanel):
        if (0 < len(self.m_LearnerFundamentalSamples)):
            self.m_PlotBayesactSim.replotOnPanel(iPlotEPAPanel)


    def clearPlots(self):
        self.m_PlotBayesactSim.clearPlots()


    def setFundamentals(self, iLearnerFundamentalSamples=None, iSimulatorFundamentalSamples=None):
        if (None != iLearnerFundamentalSamples):
            self.m_LearnerFundamentalSamples = iLearnerFundamentalSamples
        if (None != iSimulatorFundamentalSamples):
            self.m_SimulatorFundamentalSamples = iSimulatorFundamentalSamples


    def setTau(self, iLearnerTauSamples=None, iSimulatorTauSamples=None):
        if (None != iLearnerTauSamples):
            self.m_LearnerTauSamples = iLearnerTauSamples
        if (None != iSimulatorTauSamples):
            self.m_SimulatorTauSamples = iSimulatorTauSamples
