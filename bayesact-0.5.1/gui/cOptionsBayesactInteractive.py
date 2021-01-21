import wx
from cConstants import cBayesactSimConstants, cOptionSimConstants
from cNumericValidatorTextBox import cNumericValidatorTextBox
from cPlotBayesactThread import cPlotBayesactThread
from cOptionsBayesactSim import cOptionsBayesactSimPanel

class cOptionsBayesactInteractivePanel(cOptionsBayesactSimPanel):
    # The parent here is the cGuiTabs, which holds the gui itself and the options too
    def __init__(self, parent, iBayesactSim, iOptionsAgentPanel, **kwargs):
        cOptionsBayesactSimPanel.__init__(self, parent,  iBayesactSim, iOptionsAgentPanel, **kwargs)

        self.m_ClientKnowledgeStaticText.Destroy()
        self.m_ClientKnowledgeChoice.Destroy()

        self.m_AgentKnowledgeStaticText.SetPosition((100, 110))
        self.m_AgentKnowledgeChoice.SetPosition((270, 108))

        self.m_RougheningNoiseStaticText.SetPosition((100, 140))
        self.m_RougheningNoiseTextBox.SetPosition((270, 138))



    def updateSettingsFromBayesact(self):
        self.m_NumberOfSamplesTextBox.SetValue(str(self.m_BayesactSim.num_samples))
        self.m_AgentKnowledgeChoice.SetStringSelection(str(self.m_BayesactSim.agent_knowledge))
        self.m_RougheningNoiseTextBox.SetValue(str(self.m_BayesactSim.roughening_noise))


    def updateBayesactFromSettings(self):
        self.m_BayesactSim.client_gender = str(self.m_OptionsAgentPanel.m_ClientGenderChoice.GetStringSelection())
        self.m_BayesactSim.agent_gender = str(self.m_OptionsAgentPanel.m_AgentGenderChoice.GetStringSelection())
        self.m_BayesactSim.client_id_label = str(self.m_OptionsAgentPanel.m_ClientIdentityTextBox.GetValue())
        self.m_BayesactSim.agent_id_label = str(self.m_OptionsAgentPanel.m_AgentIdentityTextBox.GetValue())

        self.m_BayesactSim.num_samples = int(self.m_NumberOfSamplesTextBox.GetValue())
        self.m_BayesactSim.agent_knowledge = int(self.m_AgentKnowledgeChoice.GetStringSelection())
        self.m_BayesactSim.roughening_noise = float(self.m_RougheningNoiseTextBox.GetValue())


    def disableStartingOptions(self):
        self.m_NumberOfSamplesTextBox.Enable(False)
        self.m_AgentKnowledgeChoice.Enable(False)
        self.m_RougheningNoiseTextBox.Enable(False)


    def enableStartingOptions(self):
        self.m_NumberOfSamplesTextBox.Enable(True)
        self.m_AgentKnowledgeChoice.Enable(True)
        self.m_RougheningNoiseTextBox.Enable(True)
