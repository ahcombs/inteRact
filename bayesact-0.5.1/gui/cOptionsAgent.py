import wx
from cEPAListBox import cEPAListBox
#from cEnum importeIdentityParse
from cConstants import cInstitutionsConstants, cOptionsAgentConstants, cDataFilesConstants, cSystemConstants


# This is the panel where you define the interactants
# From what I have read, I believe the agent is the learner while the client is the simulator
# They also seem to take turns being simulator and learner which makes the simulation confusing
class cOptionsAgentPanel(wx.Panel):
    def __init__(self, parent, **kwargs):
        wx.Panel.__init__(self, parent, **kwargs)

        self.m_TextBoxSize = wx.DefaultSize
        self.m_ComboBoxSize = (110, 26)


        self.m_IdenditiesText = wx.StaticText(self, -1, "Identities", pos=(10, 10))
        self.m_IdentitiesListBox = cEPAListBox(self, iDataFile=cDataFilesConstants.m_fidentities, pos=(10, 40), size=(190, 468))

        self.m_SettingsText = wx.StaticText(self, -1, "Settings", pos=(220, 10))
        self.m_SettingsListBox = cEPAListBox(self, iDataFile=cDataFilesConstants.m_fsettings, pos=(220, 40), size=(190, 468))

        self.m_ModifiersText = wx.StaticText(self, -1, "Modifiers", pos=(430, 10))
        self.m_ModifiersListBox = cEPAListBox(self, iDataFile=cDataFilesConstants.m_fmodifiers, pos=(430, 40), size=(190, 468))


        self.m_GenderText = wx.StaticText(self, -1, "Gender", pos=(650, 35))
        self.m_GenderChoice = wx.ComboBox(self, -1, pos=(650, 65), size=self.m_ComboBoxSize,
                                          choices=cInstitutionsConstants.m_Gender,
                                          style=wx.CHOICEDLG_STYLE, value=cInstitutionsConstants.m_Gender[0])
        self.m_GenderChoice.Bind(wx.EVT_COMBOBOX, self.onSelectInstitution)


        self.m_InstitutionText = wx.StaticText(self, -1, "Institution", pos=(650, 135))
        self.m_InstitutionChoice = wx.ComboBox(self, -1, pos=(650, 165), size=self.m_ComboBoxSize,
                                               choices=cInstitutionsConstants.m_Institution,
                                               style=wx.CHOICEDLG_STYLE, value=cInstitutionsConstants.m_Institution[0])
        self.m_InstitutionChoice.Bind(wx.EVT_COMBOBOX, self.onSelectInstitution)

        textBoxXDiscrepancy = 0
        macButtonYDiscrepancy = 0
        if (cSystemConstants.m_MacOS == cSystemConstants.m_OS):
            textBoxXDiscrepancy = 3
            macButtonYDiscrepancy = -5


        self.m_ClientIdentityText = wx.StaticText(self, -1, "Client Identity", pos=(20, 510))
        self.m_ClientIdentityTextBox = wx.TextCtrl(self, pos=(17+textBoxXDiscrepancy, 533), size=(190, 26),
                                                  style=wx.TE_READONLY)

        self.m_ClientSettingText = wx.StaticText(self, -1, "Client Setting", pos=(230, 510))
        self.m_ClientSettingTextBox = wx.TextCtrl(self, pos=(227+textBoxXDiscrepancy, 533), size=(190, 26),
                                                  style=wx.TE_READONLY)

        self.m_ClientModifierText = wx.StaticText(self, -1, "Client Modifier", pos=(440, 510))
        self.m_ClientModifierTextBox = wx.TextCtrl(self, pos=(437+textBoxXDiscrepancy, 533), size=(190, 26),
                                                  style=wx.TE_READONLY)

        self.m_ClientGenderText = wx.StaticText(self, -1, "Client Gender", pos=(650, 505))
        self.m_ClientGenderChoice = wx.ComboBox(self, -1, pos=(650, 528), size=self.m_ComboBoxSize,
                                                choices=cOptionsAgentConstants.m_GenderChoices,
                                                style=wx.CHOICEDLG_STYLE)
        self.m_ClientGenderChoice.SetStringSelection(cOptionsAgentConstants.m_ClientGenderDefault)

        self.m_SetIdentityText = self.m_SetClientIdentityButton = wx.Button(self, label="Set Client", pos=(770, 530+macButtonYDiscrepancy), size=(100, 28))
        self.m_SetClientIdentityButton.Bind(wx.EVT_BUTTON, self.onSetClient)




        # Agent Identity and gender
        self.m_AgentIdentityText = wx.StaticText(self, -1, "Agent Identity", pos=(20, 570))
        self.m_AgentIdentityTextBox = wx.TextCtrl(self, pos=(17+textBoxXDiscrepancy,593), size=(190, 26),
                                                  style=wx.TE_READONLY)

        self.m_AgentSettingText = wx.StaticText(self, -1, "Agent Setting", pos=(230, 570))
        self.m_AgentSettingTextBox = wx.TextCtrl(self, pos=(227+textBoxXDiscrepancy,593), size=(190, 26),
                                                  style=wx.TE_READONLY)

        self.m_AgentModifierText = wx.StaticText(self, -1, "Agent Modifier", pos=(440, 570))
        self.m_AgentModifierTextBox = wx.TextCtrl(self, pos=(437+textBoxXDiscrepancy,593), size=(190, 26),
                                                  style=wx.TE_READONLY)

        self.m_AgentGenderText = wx.StaticText(self, -1, "Agent Gender", pos=(650, 570))
        self.m_AgentGenderChoice = wx.ComboBox(self, -1, pos=(650, 593), size=self.m_ComboBoxSize,
                                               choices=cOptionsAgentConstants.m_GenderChoices,
                                               style=wx.CHOICEDLG_STYLE)
        self.m_AgentGenderChoice.SetStringSelection(cOptionsAgentConstants.m_AgentGenderDefault)

        self.m_SetAgentIdentityButton = wx.Button(self, label="Set Agent", pos=(770, 590+macButtonYDiscrepancy), size=(100, 28))
        self.m_SetAgentIdentityButton.Bind(wx.EVT_BUTTON, self.onSetAgent)



    def onSetClient(self, iEvent):
        self.m_ClientIdentityTextBox.SetValue(self.m_IdentitiesListBox.m_SelectedIdentity)
        self.m_ClientSettingTextBox.SetValue(self.m_SettingsListBox.m_SelectedIdentity)
        self.m_ClientModifierTextBox.SetValue(self.m_ModifiersListBox.m_SelectedIdentity)


    def onSetAgent(self, iEvent):
        self.m_AgentIdentityTextBox.SetValue(self.m_IdentitiesListBox.m_SelectedIdentity)
        self.m_AgentSettingTextBox.SetValue(self.m_SettingsListBox.m_SelectedIdentity)
        self.m_AgentModifierTextBox.SetValue(self.m_ModifiersListBox.m_SelectedIdentity)


    def onSelectInstitution(self, iEvent):
        self.m_IdentitiesListBox.filterInstitution(self.m_GenderChoice.GetSelection(),
                                                 self.m_InstitutionChoice.GetSelection())

        self.m_SettingsListBox.filterInstitution(self.m_GenderChoice.GetSelection(),
                                                 self.m_InstitutionChoice.GetSelection())

        self.m_ModifiersListBox.filterInstitution(self.m_GenderChoice.GetSelection(),
                                                 self.m_InstitutionChoice.GetSelection())




class cDefaultFrame(wx.Frame):
    def __init__(self, parent, **kwargs):
        wx.Frame.__init__(self, parent, **kwargs)
        self.setAgentPanel1 = cOptionsAgentPanel(self, style=wx.NO_BORDER, pos=(0, 0), size=(1000, 700))

if __name__ == "__main__":
    app = wx.App(redirect=False)
    frame = cDefaultFrame(None, title="Title", size=(1000, 700))
    frame.Show()
    app.MainLoop()
