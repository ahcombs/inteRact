from cOptionsAgent import cOptionsAgentPanel
from cBayesactSimGui import cBayesactSimGuiPanel
from cBayesactInteractiveGui import cBayesactInteractiveGuiPanel
from cPlotBayesactThread import cPlotBayesactThread
from cConstants import cOptionSimConstants, cEPAConstants, cSystemConstants, cBoundaries
from cEnum import eEPA
import cPlotEPA2D
import wx
import sys


class cDefaultPanel(wx.Panel):
    def __init__(self, parent, iBayesactSim, iOptionsAgentPanel, iOptionsBayesactPanel, **kwargs):
        wx.Panel.__init__(self, parent, **kwargs)

        self.m_OptionsAgentPanel = iOptionsAgentPanel
        self.m_fidentitiesMale = None
        self.m_fidentitiesFemale = None
        
        

        self.m_PlotXAlign = 450
        self.m_ComboBoxXDiscrepancy = 0
        if (cSystemConstants.m_MacOS == cSystemConstants.m_OS):
            self.m_PlotXAlign = 500
            self.m_ComboBoxXDiscrepancy = 50


        # This right half of the panel holds the two plot panels for fundamental and tau
        # It is done like this so that you only draw on 2 panels instead of 4
        # These are plot panels
        self.m_PlotEPAPanel2D_Fundamental = cPlotEPA2D.cPlotPanel(self,
                                                                  iXAxisItem=cOptionSimConstants.m_XAxisDefaultKey,
                                                                  iYAxisItem=cOptionSimConstants.m_YAxisDefaultKey,
                                                                  iPlotType=eEPA.fundamental,
                                                                  pos=(self.m_PlotXAlign, 10), size=(500, 275))
        self.m_PlotEPAPanel2D_Fundamental.m_Title = "Fundamentals"


        self.m_PlotEPAPanel2D_Tau = cPlotEPA2D.cPlotPanel(self,
                                                          iXAxisItem=cOptionSimConstants.m_XAxisDefaultKey,
                                                          iYAxisItem=cOptionSimConstants.m_YAxisDefaultKey,
                                                          iPlotType=eEPA.tau,
                                                          pos=(self.m_PlotXAlign, 335), size=(500, 275))
        self.m_PlotEPAPanel2D_Tau.m_Title = "Tau"

        self.m_PlotEPAPanel2D_Fundamental.m_TwinPlots.append(self.m_PlotEPAPanel2D_Tau)
        self.m_PlotEPAPanel2D_Tau.m_TwinPlots.append(self.m_PlotEPAPanel2D_Fundamental)

        self.m_ChangeXAxisStaticText = wx.StaticText(self, -1, "X Axis", pos=(530 + self.m_ComboBoxXDiscrepancy, 300))
        self.m_ChangeXAxisComboBox = wx.ComboBox(self, -1, value=cOptionSimConstants.m_XAxisDefault,
                                                 pos=(self.m_ChangeXAxisStaticText.GetPosition()[0] + 50, self.m_ChangeXAxisStaticText.GetPosition()[1] - 5),
                                                 size=wx.DefaultSize, choices=cEPAConstants.m_EPALabels, style=wx.TE_READONLY)
        self.m_ChangeXAxisComboBox.Bind(wx.EVT_COMBOBOX, self.onChangeXAxis)

        self.m_ChangeXAxisStaticText = wx.StaticText(self, -1, "Y Axis", pos=(710 + self.m_ComboBoxXDiscrepancy, 300), size=self.m_ChangeXAxisStaticText.GetSize())
        self.m_ChangeYAxisComboBox = wx.ComboBox(self, -1, value=cOptionSimConstants.m_YAxisDefault,
                                                 pos=(self.m_ChangeXAxisStaticText.GetPosition()[0] + 50, self.m_ChangeXAxisStaticText.GetPosition()[1] - 5),
                                                 size=wx.DefaultSize, choices=cEPAConstants.m_EPALabels, style=wx.TE_READONLY)
        self.m_ChangeYAxisComboBox.Bind(wx.EVT_COMBOBOX, self.onChangeYAxis)


        self.m_Plotter = cPlotBayesactThread()

        self.m_Plotter.addPanel(self.m_PlotEPAPanel2D_Fundamental)
        self.m_Plotter.addPanel(self.m_PlotEPAPanel2D_Tau)

        iBayesactSim.plotter = self.m_Plotter

        # This is the rectangular tab you see on the left half of the sim interactive tabs
        # The style indicates that the subset of these tabs' are on the left side
        self.m_SimInteractiveTabs = cSimInteractiveTabs(self, iBayesactSim, self.m_Plotter, iOptionsAgentPanel,
                                                        iOptionsBayesactPanel, id=wx.ID_ANY, style=wx.BK_LEFT,
                                                        size=(self.m_PlotXAlign, cBoundaries.m_GlobalWindowHeight))
        self.m_Sizer = wx.BoxSizer(wx.VERTICAL)
        self.m_Sizer.Add(self.m_SimInteractiveTabs, -1)
        self.SetSizer(self.m_Sizer)
        self.Layout()
        self.Show()



    def onChangeXAxis(self, iEvent):
        value = iEvent.GetEventObject().GetValue()

        # Changes to the x or y axis of one plot will automatically be done to the other plot if it was added to the twin plot
        self.m_PlotEPAPanel2D_Fundamental.changeXAxisLabel(cEPAConstants.m_EPALabels.index(value))

        self.redrawAxes(self.m_PlotEPAPanel2D_Fundamental)
        self.redrawAxes(self.m_PlotEPAPanel2D_Tau)


    def onChangeYAxis(self, iEvent):
        value = iEvent.GetEventObject().GetValue()

        self.m_PlotEPAPanel2D_Fundamental.changeYAxisLabel(cEPAConstants.m_EPALabels.index(value))

        self.redrawAxes(self.m_PlotEPAPanel2D_Fundamental)
        self.redrawAxes(self.m_PlotEPAPanel2D_Tau)


    def redrawAxes(self, iPlotEPAPanel):
        self.m_SimInteractiveTabs.m_Plotter.replotOnPanel(iPlotEPAPanel)


# This class inherits from wx.Notebook. I just simply added panels to it
class cSimInteractiveTabs(wx.Notebook):
    def __init__(self, parent, iBayesactSim, iPlotter, iOptionsAgentPanel, iOptionsBayesactPanel, **kwargs):
        wx.Notebook.__init__(self, parent, **kwargs)

        # Both the bayesact sim and interactive should share the same BayesactSimThread and plotter
        self.m_BayesactSimThread = None
        self.m_Plotter = iPlotter

        # bayesactsim tab
        self.m_BayesactSimGuiPanel = cBayesactSimGuiPanel(self, iBayesactSim, iOptionsBayesactPanel)

        # bayeact interactive tab
        self.m_BayesactInteractiveGuiPanel = cBayesactInteractiveGuiPanel(self, iBayesactSim, iOptionsBayesactPanel, self.m_BayesactSimGuiPanel)
        self.m_BayesactSimGuiPanel.m_BayesactInteractiveGuiPanel = self.m_BayesactInteractiveGuiPanel

        iBayesactSim.sim_gui = self.m_BayesactSimGuiPanel
        iBayesactSim.interactive_gui = self.m_BayesactInteractiveGuiPanel

        self.AddPage(self.m_BayesactSimGuiPanel, "Simulator")
        self.AddPage(self.m_BayesactInteractiveGuiPanel, "Interactive")

        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.onPageChanged)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGING, self.onPageChanging)

        # To get rid of an issue with a little black box appearing on the top left of the window in MSW
        self.SendSizeEvent()


    def onPageChanged(self, event):
        #old = event.GetOldSelection()
        #new = event.GetSelection()
        #sel = self.GetSelection()
        #print 'OnPageChanged,  old:%d, new:%d, sel:%d\n' % (old, new, sel)
        event.Skip()

    def onPageChanging(self, event):
        #old = event.GetOldSelection()
        #new = event.GetSelection()
        #sel = self.GetSelection()
        #print 'OnPageChanging, old:%d, new:%d, sel:%d\n' % (old, new, sel)
        event.Skip()

