from cOptionsAgent import cOptionsAgentPanel
from cBayesactSimGui import cBayesactSimGuiPanel
from cBayesactInteractiveGui import cBayesactInteractiveGuiPanel
from cBayesactSim import cBayesactSim
from cOptionsBayesactSim import cOptionsBayesactSimPanel
from cOptionsBayesactInteractive import cOptionsBayesactInteractivePanel
from cConstants import cDataFilesConstants, cBoundaries, cSystemConstants
import cSimInteractiveTabs
import wx
import sys
import bayesact
import os


class cDefaultFrame(wx.Frame):
    def __init__(self, parent, **kwargs):
        wx.Frame.__init__(self, parent, wx.ID_ANY, "Bayesact", size=(cBoundaries.m_GlobalWindowWidth,cBoundaries.m_GlobalWindowHeight), **kwargs)

        self.m_MenuBar = wx.MenuBar()
        self.m_FileMenu = wx.Menu()

        self.m_LoadBehavioursFile = self.m_FileMenu.Append(-1, "Load Behaviours File")
        self.m_LoadIdentitiesFile = self.m_FileMenu.Append(-1, "Load Identities File")
        self.m_LoadSettingsFile = self.m_FileMenu.Append(-1, "Load Settings File")
        self.m_LoadModifiersFile = self.m_FileMenu.Append(-1, "Load Modifiers File")
        self.m_HelpFile = self.m_FileMenu.Append(-1, "&Help")


        self.m_MenuBar.Append(self.m_FileMenu, "&File")
        self.SetMenuBar(self.m_MenuBar)

        self.Bind(wx.EVT_MENU, self.onLoadBehavioursFile, self.m_LoadBehavioursFile)
        self.Bind(wx.EVT_MENU, self.onLoadIdentitiesFile, self.m_LoadIdentitiesFile)
        self.Bind(wx.EVT_MENU, self.onLoadModifiersFile, self.m_LoadModifiersFile)
        self.Bind(wx.EVT_MENU, self.onLoadSettingsFile, self.m_LoadSettingsFile)
        self.Bind(wx.EVT_MENU, self.onLoadHelpFile, self.m_HelpFile)

        self.m_Panel = wx.Panel(self)

        self.m_GuiTabs = cGuiTabs(self.m_Panel)
        self.m_Sizer = wx.BoxSizer(wx.VERTICAL)
        self.m_Sizer.Add(self.m_GuiTabs, 1, wx.ALL|wx.EXPAND, 5)
        self.m_Panel.SetSizer(self.m_Sizer)
        self.Layout()
        self.Show()


    # To get a file path
    def getFileDialog(self, iMessage, iFileExtension):
        openFileDialog = wx.FileDialog(self, iMessage, "", "",
                                       "{0} files (*.{1})|*.{1}".format(iFileExtension.upper(), iFileExtension), wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return False

        return openFileDialog.GetPath()


    def onLoadBehavioursFile(self, iEvent):
        fileName = self.getFileDialog("Open fbehaviours file", "dat")
        if (False != fileName):
            self.m_GuiTabs.m_fbehavioursFile = fileName
            self.m_GuiTabs.setSentiments()

    def onLoadIdentitiesFile(self, iEvent):
        fileName = self.getFileDialog("Open fidentities file", "dat")
        if (False != fileName):
            self.m_GuiTabs.m_fidentitiesFile = fileName
            self.m_GuiTabs.setSentiments()

    def onLoadModifiersFile(self, iEvent):
        fileName = self.getFileDialog("Open fmodifiers file", "dat")
        if (False != fileName):
            self.m_GuiTabs.m_fmodifiersFile = fileName
            self.m_GuiTabs.setSentiments()

    def onLoadSettingsFile(self, iEvent):
        fileName = self.getFileDialog("Open fsettings file", "dat")
        if (False != fileName):
            self.m_GuiTabs.m_fsettingsFile = fileName
            self.m_GuiTabs.setSentiments()

    def onLoadHelpFile(self, iEvent):
        if (cSystemConstants.m_OS == cSystemConstants.m_MacOS):
            os.system("open HelpGuide.pdf")
        elif (cSystemConstants.m_OS == cSystemConstants.m_WindowsOS):
            os.system("start HelpGuide.pdf")



class cGuiTabs(wx.Notebook):
    def __init__(self, parent):
        wx.Notebook.__init__(self, parent, id=wx.ID_ANY, style=wx.BK_TOP)

        self.m_fidentitiesFile = cDataFilesConstants.m_fidentities
        self.m_fsettingsFile = cDataFilesConstants.m_fsettings
        self.m_fmodifiersFile = cDataFilesConstants.m_fmodifiers
        self.m_fbehavioursFile = cDataFilesConstants.m_fbehaviours

        self.m_fbehavioursMale = None
        self.m_fbehavioursFemale = None

        self.m_fidentitiesMale = None
        self.m_fidentitiesFemale = None

        # Bayesact Sim
        self.m_BayesactSim = cBayesactSim()

        # Options agent tab
        self.m_OptionsAgentPanel = cOptionsAgentPanel(self)

        # Initial Settings tab for bayesact sim
        self.m_OptionsBayesactSimPanel = cOptionsBayesactSimPanel(self, self.m_BayesactSim, self.m_OptionsAgentPanel)

        # The interchangeable gui for both bayesactsim and bayesactinteractive
        self.m_SimInteractiveTabsPanel = cSimInteractiveTabs.cDefaultPanel(self, self.m_BayesactSim, self.m_OptionsAgentPanel, self.m_OptionsBayesactSimPanel)

        self.setSentiments()

        self.AddPage(self.m_OptionsAgentPanel, "Define Interactants")
        self.AddPage(self.m_OptionsBayesactSimPanel, "Bayesact Simulator Settings")
        self.AddPage(self.m_SimInteractiveTabsPanel, "Bayesact Simulator")


        # To get rid of an issue with a little black box appearing on the top left of the window in MSW
        self.SendSizeEvent()



    # To implement a change in data files, simply change the data files and call this function
    # It will go into all the objects the depend on the identities and behaviours and give them the necessary data
    # If a simulation is running, the
    def setSentiments(self):
        self.m_fbehavioursMale = bayesact.readSentiments(self.m_fbehavioursFile, "male")
        self.m_fbehavioursFemale = bayesact.readSentiments(self.m_fbehavioursFile, "female")

        self.m_fidentitiesMale = bayesact.readSentiments(self.m_fidentitiesFile, "male")
        self.m_fidentitiesFemale = bayesact.readSentiments(self.m_fidentitiesFile, "female")

        self.m_BayesactSim.identities_filename = self.m_fidentitiesFile
        self.m_BayesactSim.behaviours_filename = self.m_fbehavioursFile

        self.m_OptionsAgentPanel.m_IdentitiesListBox.refreshIdentities(self.m_fidentitiesFile)
        self.m_OptionsAgentPanel.m_SettingsListBox.refreshIdentities(self.m_fsettingsFile)
        self.m_OptionsAgentPanel.m_ModifiersListBox.refreshIdentities(self.m_fmodifiersFile)

        self.m_SimInteractiveTabsPanel.m_fidentitiesMale = self.m_fidentitiesMale
        self.m_SimInteractiveTabsPanel.m_fidentitiesFemale = self.m_fbehavioursFemale



def main(argv):
    app = wx.App(redirect=False)
    frame = cDefaultFrame(None)
    app.MainLoop()

if __name__ == "__main__":
    main(sys.argv)
