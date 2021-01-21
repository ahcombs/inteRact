import sys
import threading
sys.path.append("../")
import wx
from cBayesactSimGui import cBayesactSimGuiPanel

# To reuse the cBayesactSimGui code and add/modify a few things
class cBayesactInteractiveGuiPanel(cBayesactSimGuiPanel):
    # The parent here is the cGuiTabs, which holds the gui itself and the options too
    def __init__(self, parent, iBayesactSim, iOptionsBayesactPanel, iBayeactSimGuiPanel=None, **kwargs):
        cBayesactSimGuiPanel.__init__(self, parent, iBayesactSim, iOptionsBayesactPanel, **kwargs)

        self.m_BayesactSimGuiPanel = iBayeactSimGuiPanel

        self.m_ClientStaticText.Destroy()

        self.m_ClientAlphaStaticText.Destroy()
        self.m_ClientAlphaTextBox .Destroy()
        self.m_ClientAlphaSlider.Destroy()
        self.m_ClientAlphaButton.Destroy()

        self.m_ClientBetaOfClientStaticText.Destroy()
        self.m_ClientBetaOfClientSubscript_c_StaticText.Destroy()
        self.m_ClientBetaOfClientTextBox.Destroy()
        self.m_ClientBetaOfClientSlider.Destroy()
        self.m_ClientBetaOfClientButton.Destroy()

        self.m_ClientBetaOfAgentStaticText.Destroy()
        self.m_ClientBetaOfAgentSubscript_a_StaticText.Destroy()
        self.m_ClientBetaOfAgentTextBox.Destroy()
        self.m_ClientBetaOfAgentSlider.Destroy()
        self.m_ClientBetaOfAgentButton.Destroy()

        self.m_ClientGammaStaticText.Destroy()
        self.m_ClientGammaTextBox.Destroy()
        self.m_ClientGammaSlider.Destroy()
        self.m_ClientGammaButton.Destroy()

        self.m_AgentStaticText.Destroy()

        self.m_AgentAlphaStaticText.Destroy()
        self.m_AgentAlphaTextBox.Destroy()
        self.m_AgentAlphaSlider.Destroy()
        self.m_AgentAlphaButton.Destroy()

        self.m_AgentBetaOfClientStaticText.Destroy()
        self.m_AgentBetaOfClientSubscript_c_StaticText.Destroy()
        self.m_AgentBetaOfClientTextBox.Destroy()
        self.m_AgentBetaOfClientSlider.Destroy()
        self.m_AgentBetaOfClientButton.Destroy()

        self.m_AgentBetaOfAgentStaticText.Destroy()
        self.m_AgentBetaOfAgentSubscript_a_StaticText.Destroy()
        self.m_AgentBetaOfAgentTextBox.Destroy()
        self.m_AgentBetaOfAgentSlider.Destroy()
        self.m_AgentBetaOfAgentButton.Destroy()

        self.m_AgentGammaStaticText.Destroy()
        self.m_AgentGammaTextBox.Destroy()
        self.m_AgentGammaSlider.Destroy()
        self.m_AgentGammaButton.Destroy()

        self.m_EnvironmentNoiseStaticText.Destroy()
        self.m_EnvironmentNoiseTextBox.Destroy()
        self.m_EnvironmentNoiseSlider.Destroy()


        self.m_NumberOfStepsStaticText.Destroy()
        self.m_NumberOfStepsTextBox.Destroy()

        '''
        self.m_AgentAlphaStaticText.SetPosition((10, 30))
        self.m_AgentAlphaTextBox.SetPosition((170, 28))
        self.m_AgentAlphaSlider.SetPosition((290, 28))

        self.m_AgentBetaOfClientStaticText.SetPosition((10, 60))
        self.m_AgentBetaOfClientTextBox.SetPosition((170, 58))
        self.m_AgentBetaOfClientSlider.SetPosition((290, 58))

        self.m_AgentBetaOfAgentStaticText.SetPosition((10, 90))
        self.m_AgentBetaOfAgentTextBox.SetPosition((170, 88))
        self.m_AgentBetaOfAgentSlider.SetPosition((290, 88))

        '''

        self.m_StepButton.SetLabel("Evaluate")

        self.m_SuggestedActionsStaticText = wx.StaticText(self, -1, "Suggested Actions", pos=(10, 170))
        self.m_SuggestedActionsListBox = wx.ListBox(self, -1, pos=(10, 200), size=(190, 100),
                                                    choices=[], style=wx.LB_SINGLE)
        self.m_SuggestedActionsListBox.Bind(wx.EVT_LISTBOX, self.onSelectAction)

        self.m_ActionStaticText = wx.StaticText(self, -1, "Action", pos=(210, 170))
        self.m_ActionTextBox = wx.TextCtrl(self, -1, pos=(210, 200), size=(190, -1))

        self.m_CurrentTurnStaticText = wx.StaticText(self, -1, "Current Turn", pos=(210, 230))
        self.m_CurrentTurnTextBox = wx.TextCtrl(self, -1, pos=(210, 260), size=(190, -1), style=wx.TE_READONLY)


    def onDrawOutlines(self, iEvent=None):
        pass

    def updateBayesactFromSettings(self):
        self.m_BayesactSimGuiPanel.updateBayesactFromSettings()

    # To render these functions useless
    def onSetClientAlphaFlag(self, iEvent):
        pass

    def onSetClientBetaOfClientFlag(self, iEvent):
        pass

    def onSetClientBetaOfAgentFlag(self, iEvent):
        pass


    def onSetAgentAlphaFlag(self, iEvent):
        pass

    def onSetAgentBetaOfClientFlag(self, iEvent):
        pass

    def onSetAgentBetaOfAgentFlag(self, iEvent):
        pass


    def onSetEnvironmentNoiseFlag(self, iEvent):
        pass


    def onSetNumStepsFlag(self, iEvent):
        pass



    # You may directly set the variables in the thread anytime the thread event is cleared
    # You continue the thread by setting the thread event
    def onSelectAction(self, iEvent):
        action = str(iEvent.GetEventObject().GetStringSelection())
        self.m_ActionTextBox.SetValue(action)


    # Just simply continue the thread and set the client action
    def onStepBayesactSim(self, iEvent):
        if (None != self.m_SimInteractiveTabs.m_BayesactSimThread):
            self.m_BayesactSim.evaluate_interaction = True
            self.m_BayesactSim.thread_event.set()
        else:
            self.onStartBayesactSim(None)


    def onStartBayesactSim(self, iEvent):
        self.m_BayesactSimGuiPanel.onStartBayesactSim(None)
