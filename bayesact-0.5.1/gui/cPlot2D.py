from cEnum import eAxes, eRect
from cConstants import cPlotConstants, cPlot2DConstants
import cPlot
import wx
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas



class cPlotFrame(cPlot.cPlotFrame):

    def __init__(self, iParent, **kwargs):
        cPlot.cPlotFrame.__init__(self, iParent, **kwargs)

    def initPanel(self, **kwargs):
        self.m_PlotPanel = cPlotPanel(self, **kwargs)


class cPlotPanel(cPlot.cPlotPanel):

    def __init__(self, iParent, **kwargs):
        cPlot.cPlotPanel.__init__(self, iParent, **kwargs)

        self.m_NumAxes = 2

        self.m_Figure = Figure(figsize=self.getFigSize(),
            facecolor=cPlotConstants.m_BackgroundColour, edgecolor=cPlotConstants.m_BackgroundColour)


        self.m_Axes = self.m_Figure.add_axes(cPlot2DConstants.m_Rect)
        self.m_Axes.set_xlim(self.m_XAxisMin, self.m_XAxisMax)
        self.m_Axes.set_ylim(self.m_YAxisMin, self.m_YAxisMax)

        '''
        # Details of m_Rect, is described in cConstants.cPlot2DConstants
        # This will be used for pretty panning, that is, panning the plot and making it look nice
        # I will need to calculate the optimal stepping for the pixels
        rect = cPlot2DConstants.m_Rect# [0.19, 0.13, 0.8, 0.79]
        pixelLength_X_Axis = rect[eRect.fractionOfX] * self.GetSize()[0]
        pixelLength_Y_Axis = rect[eRect.fractionOfY] * self.GetSize()[1]


        self.m_OptimalXStep = 2.0 / pixelLength_X_Axis
        self.m_OptimalYStep = 1.0 / pixelLength_Y_Axis
        '''

        self.m_Canvas = FigureCanvas(self, -1, self.m_Figure)

        # Enables interactivity
        self.m_Canvas.mpl_connect("motion_notify_event", self.onMouseMove)
        self.m_Canvas.mpl_connect("button_press_event", self.onMousePress)
        self.m_Canvas.mpl_connect("button_release_event", self.onMouseRelease)
        self.m_Canvas.mpl_connect("key_press_event", self.onKeyPress)
        self.m_Canvas.mpl_connect("scroll_event", self.onScroll)


    def getFigSize(self):
        x, y = self.GetSize()
        x = x * cPlot2DConstants.m_FigRatioX
        y = y * cPlot2DConstants.m_FigRatioY
        return (x, y)

    def plotScatter(self, iXData, iYData, iAutoScaling=False, iRedraw=False, iUpdate=True, **kwargs):
        if (True == iRedraw):
            self.clearAxes()
        if (False == iAutoScaling):
            tempXAxis = list(self.m_Axes.get_xlim())
            tempYAxis = list(self.m_Axes.get_ylim())
            self.m_Axes.scatter(iXData, iYData, **kwargs)
            self.m_Axes.set_xlim(tempXAxis)
            self.m_Axes.set_ylim(tempYAxis)
        else:
            self.m_Axes.scatter(iXData, iYData, **kwargs)

        if (True == iUpdate):
            self.redrawAxes()


    def resetAxes(self):
        self.m_Axes.set_xlim(cPlotConstants.m_DefaultXAxisMin, cPlotConstants.m_DefaultXAxisMax)
        self.m_Axes.set_ylim(cPlotConstants.m_DefaultYAxisMin, cPlotConstants.m_DefaultXAxisMax)
        self.updateAxesData()
        self.redrawAxes()


    def updateAxesData(self):
        self.m_XAxisMin, self.m_XAxisMax = self.m_Axes.get_xlim()
        self.m_YAxisMin, self.m_YAxisMax = self.m_Axes.get_ylim()

        self.m_XAxisLength = (self.m_XAxisMin - self.m_XAxisMax)
        self.m_YAxisLength = (self.m_YAxisMin - self.m_YAxisMax)


    def onMousePress(self, iEvent):
        if (iEvent.inaxes == self.m_Axes):
            self.m_PreviousMouseX, self.m_PreviousMouseY = iEvent.xdata, iEvent.ydata
            self.m_PreviousMouseXPixel, self.m_PreviousMouseYPixel = iEvent.x, iEvent.y


    # modified from mpl.toolkits.mplot3d.axes3d._on_move
    def onMouseMove(self, iEvent):
        if (not iEvent.button):
            return

        currentMouseX, currentMouseY = iEvent.xdata, iEvent.ydata
        currentMouseXPixel, currentMouseYPixel = iEvent.x, iEvent.y

        # In case the mouse is out of bounds.
        if (currentMouseX == None):
            return

        #diffMouseX = (currentMouseX - self.m_PreviousMouseX) * cPlot2DConstants.m_MouseDragSensitivity
        #diffMouseY = (currentMouseY - self.m_PreviousMouseY) * cPlot2DConstants.m_MouseDragSensitivity

        # panning
        # 3 represents right click
        if (cPlotConstants.m_MousePanButton == iEvent.button):
            self.updateAxesData()

            #diffMouseX *= cPlot2DConstants.m_PanSensitivity
            #diffMouseY *= cPlot2DConstants.m_PanSensitivity

            diffMouseX = currentMouseX - self.m_PreviousMouseX
            diffMouseY = currentMouseY - self.m_PreviousMouseY

            diffMouseXPixel = currentMouseXPixel - self.m_PreviousMouseXPixel
            diffMouseYPixel = currentMouseYPixel - self.m_PreviousMouseYPixel

            lengthX = abs(self.m_XAxisMax - self.m_XAxisMin)
            lengthY = abs(self.m_YAxisMax - self.m_YAxisMin)


            if (False == self.m_LockAxes[eAxes.xAxis]):
                if (1 <= abs(diffMouseXPixel)):
                    shiftedX = diffMouseX * (1 / lengthX)
                    self.shiftXAxis(-shiftedX)

            if (False == self.m_LockAxes[eAxes.yAxis]):
                if (1 <= abs(diffMouseYPixel)):
                    shiftedY = diffMouseY * (1 / lengthY)
                    self.shiftYAxis(-shiftedY)

            self.redrawAxes()


    def zoomAxes(self, iZoomAmount):
        self.updateAxesData()

        diffX = (self.m_XAxisMax - self.m_XAxisMin) * (0.1 * iZoomAmount)
        diffY = (self.m_YAxisMax - self.m_YAxisMin) * (0.1 * iZoomAmount)

        self.m_Axes.set_xlim(self.m_XAxisMin + diffX, self.m_XAxisMax - diffX)
        self.m_Axes.set_ylim(self.m_YAxisMin + diffY, self.m_YAxisMax - diffY)

        self.redrawAxes()

    def rescaleAxes(self):
        # Recompute bounds
        self.m_Axes.relim()
        self.m_Axes.autoscale_view()
