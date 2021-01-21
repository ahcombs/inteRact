from cConstants import cEPAConstants
from cEnum import eEPA
import cPlot2D
import cPlot3D


class cPlotFrame(cPlot2D.cPlotFrame):
    def __init__(self, iParent, **kwargs):
        cPlot2D.cPlotFrame.__init__(self, iParent, **kwargs)

    def initPanel(self, **kwargs):
        self.m_PlotPanel = cPlotPanel(self, **kwargs)


class cPlotPanel2D(cPlot2D.cPlotPanel):

    def __init__(self, iParent, iXAxisItem=eEPA.evaluation, iYAxisItem=eEPA.potency, iZAxisItem=eEPA.activity, **kwargs):
        cPlot2D.cPlotPanel.__init__(self, iParent, **kwargs)

        self.m_Title = ""
        
        self.m_XAxisItem = iXAxisItem
        self.m_YAxisItem = iYAxisItem
        self.m_ZAxisItem = iZAxisItem

        
    def getSentimentEPAIndex(self, iEPA, iSentiment):
        return iEPA + (cEPAConstants.m_Dimensions * iSentiment)


    # Axis items are the enumerations of the elements in eEPA, so they're basically numbers
    def setAxis(iXAxisItem, iYAxisItem, iZAxisItem=None):
        self.m_XAxisItem = iXAxisItem
        self.m_YAxisItem = iYAxisItem
        if (None != iZAxisItem):
            self.m_ZAxisItem = iZAxisItem
