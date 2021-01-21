import wx
from cConstants import cBayesactSimConstants, cOptionSimConstants, cSystemConstants
from cNumericValidatorTextBox import cNumericValidatorTextBox


class cOptionsBayesactSimPanel(wx.Panel):
    # The parent here is the cGuiTabs, which holds the gui itself and the options too
    def __init__(self, parent, iBayesactSim, iOptionsAgentPanel, **kwargs):
        wx.Panel.__init__(self, parent, **kwargs)

        self.m_BayesactSim = iBayesactSim
        self.m_OptionsAgentPanel = iOptionsAgentPanel

        self.m_BayesactSimThread = None

        macButtonYDiscrepancy = 0
        if (cSystemConstants.m_OS == cSystemConstants.m_MacOS):
            macButtonYDiscrepancy = -5


        # These are for all the options you can fill into the simulation
        ########################################################################################
        self.m_TextBoxSize = wx.DefaultSize
        self.m_ComboBoxSize = wx.DefaultSize

        self.m_RandomSeedStaticText = wx.StaticText(self, -1, "Random Seed", pos=(100, 50))
        self.m_RandomSeedTextBox = wx.TextCtrl(self, -1, pos=(270, 48), size=self.m_TextBoxSize,
                                               value=str(self.m_BayesactSim.rseed),
                                               validator=cNumericValidatorTextBox(iDecimals=False, iNegative=False))
        self.m_RandomSeedButton = wx.Button(self, -1, "?", pos=(70, 50+macButtonYDiscrepancy), size=(20, 20))
        self.m_RandomSeedButton.Bind(wx.EVT_BUTTON, lambda event, message="Copy this down to repeat simulations", title="Help": self.onSpawnMessageBox(event, message, title))


        self.m_NumberOfSamplesStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_NumberOfSamples, pos=(100, 80))
        self.m_NumberOfSamplesTextBox = wx.TextCtrl(self, -1, pos=(270, 78), size=self.m_TextBoxSize,
                                                   value=str(cOptionSimConstants.m_NumberOfSamplesDefault),
                                                   validator=cNumericValidatorTextBox(iDecimals=False, iNegative=False))
        self.m_NumberOfSamplesTextBox.Bind(wx.EVT_TEXT, self.onSetNumSamples)
        self.m_NumberOfSamplesButton = wx.Button(self, -1, "?", pos=(70, 80+macButtonYDiscrepancy), size=(20, 20))
        self.m_NumberOfSamplesButton.Bind(wx.EVT_BUTTON, lambda event, message="Number of Samples to experiment\nThe maximum number of samples to show on plots is set in constants", title="Help": self.onSpawnMessageBox(event, message, title))



        self.m_ClientKnowledgeStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_ClientKnowledge, pos=(100, 110))
        self.m_ClientKnowledgeChoice = wx.ComboBox(self, -1, pos=(270, 108), size=self.m_ComboBoxSize,
                                                   choices=cOptionSimConstants.m_KnowledgeChoices,
                                                   style=wx.CHOICEDLG_STYLE)
        self.m_ClientKnowledgeChoice.SetStringSelection(cOptionSimConstants.m_ClientKnowledgeDefault)
        self.m_ClientKnowledgeButton = wx.Button(self, -1, "?", pos=(70, 110+macButtonYDiscrepancy), size=(20, 20))
        self.m_ClientKnowledgeButton.Bind(wx.EVT_BUTTON, lambda event, message="Client Knowledge:\n0: knows own id, not client id\n1: knows own id, knows client id is one of the num_confusers possibilities\n2: knows own id, knows client id", title="Help": self.onSpawnMessageBox(event, message, title))

        self.m_AgentKnowledgeStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_AgentKnowledge, pos=(100, 140))
        self.m_AgentKnowledgeChoice = wx.ComboBox(self, -1, pos=(270, 138), size=self.m_ComboBoxSize,
                                                  choices=cOptionSimConstants.m_KnowledgeChoices,
                                                  style=wx.CHOICEDLG_STYLE)
        self.m_AgentKnowledgeChoice.SetStringSelection(cOptionSimConstants.m_AgentKnowledgeDefault)
        self.m_AgentKnowledgeButton = wx.Button(self, -1, "?", pos=(70, 140+macButtonYDiscrepancy), size=(20, 20))
        self.m_AgentKnowledgeButton.Bind(wx.EVT_BUTTON, lambda event, message="Agent Knowledge:\n0: knows own id, not client id\n1: knows own id, knows client id is one of the num_confusers possibilities\n2: knows own id, knows client id", title="Help": self.onSpawnMessageBox(event, message, title))


        self.m_RougheningNoiseStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_RougheningNoise, pos=(100, 170))
        self.m_RougheningNoiseTextBox = wx.TextCtrl(self, -1, pos=(270, 168), size=self.m_TextBoxSize,
                                                    value=str(cOptionSimConstants.m_RougheningNoiseDefault),
                                                    validator=cNumericValidatorTextBox(iDecimals=True, iNegative=True))
        self.m_RougheningNoiseButton = wx.Button(self, -1, "?", pos=(70, 170+macButtonYDiscrepancy), size=(20, 20))
        self.m_RougheningNoiseButton.Bind(wx.EVT_BUTTON, lambda event, message="Automatically set by number of samples\nYou may manually adjust this", title="Help": self.onSpawnMessageBox(event, message, title))


        self.m_MimicInteractStaticText = wx.StaticText(self, -1, "Mimic Interact", pos=(100, 200))
        self.m_MimicInteractCheckBox = wx.CheckBox(self, -1, pos=(270, 200))
        self.m_MimicInteractCheckBox.Bind(wx.EVT_CHECKBOX, self.onCheckMimicInteract)


        ########################################################################################

    # To set the values of the gui to the values in bayesact
    def updateSettingsFromBayesact(self):
        self.m_RandomSeedTextBox.SetValue(str(self.m_BayesactSim.rseed))

        self.m_NumberOfSamplesTextBox.SetValue(str(self.m_BayesactSim.num_samples))
        self.m_ClientKnowledgeChoice.SetStringSelection(str(self.m_BayesactSim.client_knowledge))
        self.m_AgentKnowledgeChoice.SetStringSelection(str(self.m_BayesactSim.agent_knowledge))

        self.m_RougheningNoiseTextBox.SetValue(str(self.m_BayesactSim.roughening_noise))


    # To set the values of bayesact to the values in the gui
    # Should only be used to initialize
    def updateBayesactFromSettings(self):
        self.m_BayesactSim.rseed = int(self.m_RandomSeedTextBox.GetValue())

        self.m_BayesactSim.client_gender = str(self.m_OptionsAgentPanel.m_ClientGenderChoice.GetStringSelection())
        self.m_BayesactSim.agent_gender = str(self.m_OptionsAgentPanel.m_AgentGenderChoice.GetStringSelection())
        self.m_BayesactSim.client_id_label = str(self.m_OptionsAgentPanel.m_ClientIdentityTextBox.GetValue())
        self.m_BayesactSim.agent_id_label = str(self.m_OptionsAgentPanel.m_AgentIdentityTextBox.GetValue())

        self.m_BayesactSim.num_samples = int(self.m_NumberOfSamplesTextBox.GetValue())

        self.m_BayesactSim.client_knowledge = int(self.m_ClientKnowledgeChoice.GetStringSelection())
        self.m_BayesactSim.agent_knowledge = int(self.m_AgentKnowledgeChoice.GetStringSelection())

        self.m_BayesactSim.roughening_noise = float(self.m_RougheningNoiseTextBox.GetValue())

        self.m_BayesactSim.mimic_interact = self.m_MimicInteractCheckBox.IsChecked()


    def disableStartingOptions(self):
        self.m_RandomSeedTextBox.Enable(False)
        self.m_NumberOfSamplesTextBox.Enable(False)

        self.m_ClientKnowledgeChoice.Enable(False)
        self.m_AgentKnowledgeChoice.Enable(False)

        self.m_RougheningNoiseTextBox.Enable(False)
        self.m_MimicInteractCheckBox.Enable(False)


    def enableStartingOptions(self):
        self.m_RandomSeedTextBox.Enable(True)
        self.m_NumberOfSamplesTextBox.Enable(True)

        self.m_ClientKnowledgeChoice.Enable(True)
        self.m_AgentKnowledgeChoice.Enable(True)

        self.m_RougheningNoiseTextBox.Enable(True)
        self.m_MimicInteractCheckBox.Enable(True)

    def onSpawnMessageBox(self, iEvent, iMessage, iTitle):
        wx.MessageBox(iMessage, iTitle)


    # To set the roughening noise based on samples, you can still change it as long as you change it last
    def onSetNumSamples(self, iEvent):
        numSamples = self.m_NumberOfSamplesTextBox.GetValue()
        if (numSamples.replace(".", "", 1).isdigit()):
            self.m_RougheningNoiseTextBox.SetValue(str(int(numSamples)**(-1.0/3.0)))


    def onCheckMimicInteract(self, iEvent):
        self.m_BayesactSim.mimic_interact = iEvent.GetEventObject().IsChecked()
