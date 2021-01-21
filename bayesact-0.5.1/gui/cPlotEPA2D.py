from cConstants import cEPAConstants, cPlotConstants
from cEnum import eEPA
import cPlot2D
import cPlotEPA
import sys
sys.path.append("../")
import bayesact
import wx


class cPlotFrame(cPlotEPA.cPlotFrame):
    def __init__(self, iParent, **kwargs):
        cPlot2D.cPlotFrame.__init__(self, iParent, **kwargs)

    def initPanel(self, *args, **kwargs):
        self.m_PlotPanel = cPlotPanel(self, **kwargs)


class cPlotPanel(cPlot2D.cPlotPanel):

    def __init__(self, iParent, iXAxisItem=eEPA.evaluation, iYAxisItem=eEPA.potency, iPlotType=eEPA.fundamental, **kwargs):
        cPlot2D.cPlotPanel.__init__(self, iParent, **kwargs)

        self.m_XAxisItem = iXAxisItem
        self.m_YAxisItem = iYAxisItem

        self.m_PlotType = iPlotType

        self.m_SimInteractiveTabsPanel = iParent

        # The other plots, changes the x and y boundaries of this plot will be done the same to other plots
        # Good for comparing multiple plots
        self.m_TwinPlots = []

        self.m_LearnerSamples = []
        self.m_SimulatorSamples = []



    def getSentimentEPAIndex(self, iEPA, iSentiment):
        return iEPA + (cEPAConstants.m_Dimensions * iSentiment)


    # Axis items are the enumerations of the elements in eEPA, so they're basically numbers
    def setAxis(iXAxisItem, iYAxisItem):
        self.m_XAxisItem = iXAxisItem
        self.m_YAxisItem = iYAxisItem


    def plotEPA(self, iLearnerSamples, iSimulatorSamples, iLearnerPreviousAction, iSimulatorPreviousAction):
        self.clearAxes()
        # Size is the size of the point in terms of viewing size
        lsize=50
        # Alpha is the opacity of the point
        lalpha=0.5

        self.m_LearnerSamples = iLearnerSamples
        self.m_SimulatorSamples = iSimulatorSamples

        if (0 < len(iLearnerSamples)):
            # Learner's sentiments on self and other, green and pink respectively
            learnerSamplesXIndexSelf = self.getSentimentEPAIndex(self.m_XAxisItem, cEPAConstants.m_SelfMultiplier)
            learnerSamplesYIndexSelf = self.getSentimentEPAIndex(self.m_YAxisItem, cEPAConstants.m_SelfMultiplier)

            learnerSamplesXIndexOther = self.getSentimentEPAIndex(self.m_XAxisItem, cEPAConstants.m_OtherMultiplier)
            learnerSamplesYIndexOther = self.getSentimentEPAIndex(self.m_YAxisItem, cEPAConstants.m_OtherMultiplier)


            self.plotScatter(
                iLearnerSamples[learnerSamplesXIndexSelf],
                iLearnerSamples[learnerSamplesYIndexSelf],
                iAutoScaling=False, iRedraw=False, iUpdate=False, marker="o", s=lsize, c="cyan", alpha=lalpha, animated=False)

            self.plotScatter(
                iLearnerSamples[learnerSamplesXIndexOther],
                iLearnerSamples[learnerSamplesYIndexOther],
                iAutoScaling=False, iRedraw=False, iUpdate=False, marker="o", s=lsize, c="blue", alpha=lalpha, animated=False)

            # This also checks that when an action has an EPA rating of (0, 0, 0), it will not plot it
            if (0 < len(iLearnerPreviousAction)):
                if ((0, 0, 0) == (iLearnerPreviousAction[0], iLearnerPreviousAction[1], iLearnerPreviousAction[2])):
                    pass
                else:
                    self.plotScatter(
                        iLearnerPreviousAction[self.m_XAxisItem],
                        iLearnerPreviousAction[self.m_YAxisItem],
                        marker="*",  s=200, c="turquoise", alpha=1)

        if (0 < len(iSimulatorSamples)):
            # Simulator's sentiments on self and other, goldenrod and blue respectively
            simulatorSamplesXIndexSelf = self.getSentimentEPAIndex(self.m_XAxisItem, cEPAConstants.m_SelfMultiplier)
            simulatorSamplesYIndexSelf = self.getSentimentEPAIndex(self.m_YAxisItem, cEPAConstants.m_SelfMultiplier)

            simulatorSamplesXIndexOther = self.getSentimentEPAIndex(self.m_XAxisItem, cEPAConstants.m_OtherMultiplier)
            simulatorSamplesYIndexOther = self.getSentimentEPAIndex(self.m_YAxisItem, cEPAConstants.m_OtherMultiplier)

            self.plotScatter(
                iSimulatorSamples[simulatorSamplesXIndexSelf],
                iSimulatorSamples[simulatorSamplesYIndexSelf],
                iAutoScaling=False, iRedraw=False, iUpdate=False, marker="o", s=lsize, c="magenta", alpha=lalpha, animated=False)

            self.plotScatter(
                iSimulatorSamples[simulatorSamplesXIndexOther],
                iSimulatorSamples[simulatorSamplesYIndexOther],
                iAutoScaling=False, iRedraw=False, iUpdate=False, marker="o", s=lsize, c="red", alpha=lalpha, animated=False)

            if (0 < len(iSimulatorPreviousAction)):
                if ((0, 0, 0) == (iSimulatorPreviousAction[0],iSimulatorPreviousAction[1], iSimulatorPreviousAction[2])):
                    pass
                else:
                    self.plotScatter(
                        iSimulatorPreviousAction[self.m_XAxisItem],
                        iSimulatorPreviousAction[self.m_YAxisItem],
                        marker="*",  s=200, c="magenta", alpha=1)


        self.m_Axes.set_title(self.m_Title, fontsize=12)
        self.m_Axes.set_xlabel(cEPAConstants.m_EPALabels[self.m_XAxisItem])
        self.m_Axes.set_ylabel(cEPAConstants.m_EPALabels[self.m_YAxisItem])
        self.redrawAxes()


    def onMousePress(self, iEvent):
        # Returns (index, minDist), where minDist is the minimum euclidean distance calculated
        def getMin(data, x, y):
            index = 0
            minDist = ((data[0][0] - x) ** 2) + ((data[1][0] - y) ** 2)
            points = len(data[0])

            for i in range(points-1):
                dist = ((data[0][i+1] - x) ** 2) + ((data[1][i+1] - y) ** 2)
                if (dist < minDist):
                    minDist = dist
                    index = i+1

            return (index, minDist)

        def getSampleEPA(data, dataIndex, evaluationIndex, potencyIndex, activityIndex):
            return [data[evaluationIndex][dataIndex], data[potencyIndex][dataIndex], data[activityIndex][dataIndex]]

        # Do default function, then find closest point, if anything is plotted
        # Please note that this does not include the previous action
        super(cPlotPanel, self).onMousePress(iEvent)

        # 1 represents left click, check for closest point when left clicking
        if(1 != iEvent.button):
            return

        if (iEvent.inaxes != self.m_Axes):
            return

        if (0 >= len(self.m_LearnerSamples)):
            return

        xPoint = iEvent.xdata
        yPoint = iEvent.ydata

        learnerSamplesXIndexSelf = self.getSentimentEPAIndex(self.m_XAxisItem, cEPAConstants.m_SelfMultiplier)
        learnerSamplesYIndexSelf = self.getSentimentEPAIndex(self.m_YAxisItem, cEPAConstants.m_SelfMultiplier)
        learnerSamplesXIndexOther = self.getSentimentEPAIndex(self.m_XAxisItem, cEPAConstants.m_OtherMultiplier)
        learnerSamplesYIndexOther = self.getSentimentEPAIndex(self.m_YAxisItem, cEPAConstants.m_OtherMultiplier)

        simulatorSamplesXIndexSelf = self.getSentimentEPAIndex(self.m_XAxisItem, cEPAConstants.m_SelfMultiplier)
        simulatorSamplesYIndexSelf = self.getSentimentEPAIndex(self.m_YAxisItem, cEPAConstants.m_SelfMultiplier)
        simulatorSamplesXIndexOther = self.getSentimentEPAIndex(self.m_XAxisItem, cEPAConstants.m_OtherMultiplier)
        simulatorSamplesYIndexOther = self.getSentimentEPAIndex(self.m_YAxisItem, cEPAConstants.m_OtherMultiplier)

        # To find the closest point to where the mouse clicked
        visibleLearnerSelfData = [self.m_LearnerSamples[learnerSamplesXIndexSelf], self.m_LearnerSamples[learnerSamplesYIndexSelf]]
        visibleLearnerOtherData = [self.m_LearnerSamples[learnerSamplesXIndexOther], self.m_LearnerSamples[learnerSamplesYIndexOther]]

        visibleSimulatorSelfData = [self.m_SimulatorSamples[simulatorSamplesXIndexSelf], self.m_SimulatorSamples[simulatorSamplesYIndexSelf]]
        visibleSimulatorOtherData = [self.m_SimulatorSamples[simulatorSamplesXIndexOther], self.m_SimulatorSamples[simulatorSamplesYIndexOther]]

        learnerSelf, learnerOther, simulatorSelf, simulatorOther = range(4)

        currentMinIndex, currentMinDist = getMin(visibleLearnerSelfData, xPoint, yPoint)
        currentMinData = learnerSelf

        allOtherData = [visibleLearnerOtherData, visibleSimulatorSelfData, visibleSimulatorOtherData]

        for i in range(len(allOtherData)):
            tempMinIndex, tempMinDist = getMin(allOtherData[i], xPoint, yPoint)
            if (tempMinDist < currentMinDist):
                currentMinIndex, currentMinDist = tempMinIndex, tempMinDist
                currentMinData = i+1

        if (currentMinData == learnerSelf):
            epa = getSampleEPA(self.m_LearnerSamples, currentMinIndex, eEPA.evaluationSelf, eEPA.potencySelf, eEPA.activitySelf)
        elif (currentMinData == learnerOther):
            epa = getSampleEPA(self.m_LearnerSamples, currentMinIndex, eEPA.evaluationOther, eEPA.potencyOther, eEPA.activityOther)
        elif (currentMinData == simulatorSelf):
            epa = getSampleEPA(self.m_SimulatorSamples, currentMinIndex, eEPA.evaluationSelf, eEPA.potencySelf, eEPA.activitySelf)
        else:
            epa = getSampleEPA(self.m_SimulatorSamples, currentMinIndex, eEPA.evaluationOther, eEPA.potencyOther, eEPA.activityOther)

        gender = self.m_SimInteractiveTabsPanel.m_OptionsAgentPanel.m_ClientGenderChoice.GetStringSelection()

        if ("male" == gender):
            estimatedIdentity = bayesact.findNearestEPAVector(epa, self.m_SimInteractiveTabsPanel.m_fidentitiesMale)
        else:
            estimatedIdentity = bayesact.findNearestEPAVector(epa, self.m_SimInteractiveTabsPanel.m_fidentitiesFemale)

        # Those threes mean 3 decimal places
        message = "You clicked on point: {}".format((round(xPoint, 3), round(yPoint, 3))) +\
                  "\nHere is the closest point:" +\
                  "\nEvaluation: {}\nPotency: {}\nActivity: {}".format(round(epa[eEPA.evaluation], 3), round(epa[eEPA.potency], 3), round(epa[eEPA.activity], 3)) +\
                  "\nClosest Identity: {}".format(estimatedIdentity) +\
                  "\nType: {}".format(cEPAConstants.m_PlotDetails[currentMinData])

        wx.MessageBox(message, "Closest Point")


    def changeXAxisLabel(self, iLabel):
        self.m_XAxisItem = iLabel
        for plotEPA2D in self.m_TwinPlots:
            plotEPA2D.m_XAxisItem = iLabel

    def changeYAxisLabel(self, iLabel):
        self.m_YAxisItem = iLabel
        for plotEPA2D in self.m_TwinPlots:
            plotEPA2D.m_YAxisItem = iLabel

    def shiftXAxis(self, iShiftAmount):
        super(cPlotPanel, self).shiftXAxis(iShiftAmount)
        self.updateAxesData()
        for plotEPA2D in self.m_TwinPlots:
            plotEPA2D.m_Axes.set_xlim(self.m_XAxisMin, self.m_XAxisMax)
            plotEPA2D.redrawAxes()

    def shiftYAxis(self, iShiftAmount):
        super(cPlotPanel, self).shiftYAxis(iShiftAmount)
        self.updateAxesData()
        for plotEPA2D in self.m_TwinPlots:
            plotEPA2D.m_Axes.set_ylim(self.m_YAxisMin, self.m_YAxisMax)
            plotEPA2D.redrawAxes()

    def zoomAxes(self, iZoomAmount):
        super(cPlotPanel, self).zoomAxes(iZoomAmount)
        self.updateAxesData()
        for plotEPA2D in self.m_TwinPlots:
            plotEPA2D.m_Axes.set_xlim(self.m_XAxisMin, self.m_XAxisMax)
            plotEPA2D.m_Axes.set_ylim(self.m_YAxisMin, self.m_YAxisMax)
            plotEPA2D.redrawAxes()

    def resetAxes(self):
        super(cPlotPanel, self).resetAxes()
        self.updateAxesData()
        for plotEPA2D in self.m_TwinPlots:
            plotEPA2D.m_Axes.set_xlim(self.m_XAxisMin, self.m_XAxisMax)
            plotEPA2D.m_Axes.set_ylim(self.m_YAxisMin, self.m_YAxisMax)
            plotEPA2D.redrawAxes()
