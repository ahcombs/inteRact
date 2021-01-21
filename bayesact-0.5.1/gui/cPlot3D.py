from cEnum import eAxes
from cConstants import cPlotConstants, cPlot3DConstants
import cPlot
import wx
import numpy as NP
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.axes import Axes as mplAxes
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import mpl_toolkits.mplot3d.proj3d as proj3d


class cPlotFrame(cPlot.cPlotFrame):
    def __init__(self, iParent, **kwargs):
        cPlot.cPlotFrame.__init__(self, iParent, **kwargs)

    def initPanel(self, **kwargs):
        self.m_PlotPanel = cPlotPanel(self, **kwargs)


class cPlotPanel(cPlot.cPlotPanel):

    def __init__(self, iParent, **kwargs):
        cPlot.cPlotPanel.__init__(self, iParent, **kwargs)

        self.m_Figure = Figure()

        # In order to use the custom onMouseMove, the rotate and zoom btn must be set to nothing
        self.m_Canvas = FigureCanvas(self, -1, self.m_Figure)

        self.m_Axes = Axes3D(self.m_Figure)

        self.m_Axes.set_axisbelow(True)

        self.m_Axes._rotate_btn = []
        self.m_Axes._zoom_btn = []

        self.resetAxes()

        # Enables interactivity
        self.m_Canvas.mpl_connect("motion_notify_event", self.onMouseMove)
        self.m_Canvas.mpl_connect("button_press_event", self.onMousePress)
        self.m_Canvas.mpl_connect("button_release_event", self.onMouseRelease)
        self.m_Canvas.mpl_connect("key_press_event", self.onKeyPress)
        self.m_Canvas.mpl_connect("scroll_event", self.onScroll)


    def clearGraph(self):
        self.m_Figure.clf()

        #import pdb; pdb.set_trace()
        self.m_Canvas.draw()

        #print "\n\n\n\n\n\njdaidjasoidjasiodjas"


    def plotScatter(self, iXData, iYData, iZData=None, iAutoScaling=False, iRedraw=False, iUpdate=False, **kwargs):
        '''
        if (True == iRedraw):
            self.m_Axes.cla()

        self.m_XData = iXData
        self.m_YData = iYData
        self.m_ZData = iZData

        patch = mplAxes.scatter(self.m_Axes, iXData, iYData, **kwargs)
        if (None == iZData):
            iZata = np.zeros(len(iXData))

        art3d.patch_collection_2d_to_3d(patch, zs=iZData)

        self.addPatch(patch)
        if (True == iUpdate):
            self.redrawPlot()
        '''
        if (True == iRedraw):
            self.clearAxes()

        if (False == iAutoScaling):
            tempXAxis = list(self.m_Axes.get_xlim())
            tempYAxis = list(self.m_Axes.get_ylim())
            tempZAxis = list(self.m_Axes.get_zlim())
            self.m_Axes.scatter(iXData, iYData, iZData, **kwargs)
            self.m_Axes.set_xlim(tempXAxis)
            self.m_Axes.set_ylim(tempYAxis)
            self.m_Axes.set_zlim(tempZAxis)
        else:
            self.m_Axes.scatter(iXData, iYData, iZData, **kwargs)

        if (True == iUpdate):
            self.redrawAxes()
            #self.redrawAxes()



    def resetAxes(self):
        self.m_Axes.set_xlim(cPlotConstants.m_DefaultXAxisMin, cPlotConstants.m_DefaultXAxisMax)
        self.m_Axes.set_ylim(cPlotConstants.m_DefaultYAxisMin, cPlotConstants.m_DefaultYAxisMax)
        self.m_Axes.set_zlim(cPlotConstants.m_DefaultZAxisMin, cPlotConstants.m_DefaultZAxisMax)
        self.m_Axes.azim = cPlot3DConstants.m_DefaultAzim
        self.m_Axes.elev = cPlot3DConstants.m_DefaultElev
        self.updateAxesData()
        self.redrawAxes()


    def updateAxesData(self):
        self.m_XAxisMin, self.m_XAxisMax = self.m_Axes.get_xlim()
        self.m_YAxisMin, self.m_YAxisMax = self.m_Axes.get_ylim()
        self.m_ZAxisMin, self.m_ZAxisMax = self.m_Axes.get_zlim()

        self.m_XAxisLength = (self.m_XAxisMin - self.m_XAxisMax)
        self.m_YAxisLength = (self.m_YAxisMin - self.m_YAxisMax)
        self.m_ZAxisLength = (self.m_ZAxisMin - self.m_ZAxisMax)


    # modified from mpl.toolkits.mplot3d.axes3d._on_move
    def onMouseMove(self, iEvent):
        if (not iEvent.button):
            return

        # Getting the currentMouse X and Y must be done this way in WindowsOS
        # In MacOS, you may just simply get the x and y coordinates from iEvent.xdata and ydata
        trans = iEvent.inaxes.transData.inverted()
        currentMouseX, currentMouseY = trans.transform_point((iEvent.x, iEvent.y))

        # In case the mouse is out of bounds.
        if (currentMouseX == None):
            return

        self.updateAxesData()

        diffMouseX = (currentMouseX - self.m_PreviousMouseX) * self.m_XAxisLength * cPlot3DConstants.m_MouseDragSensitivity
        diffMouseY = (currentMouseY - self.m_PreviousMouseY) * self.m_YAxisLength * cPlot3DConstants.m_MouseDragSensitivity

        # Rotation
        if (cPlot3DConstants.m_MouseRotateButton == iEvent.button):
            self.m_PreviousMouseX, self.m_PreviousMouseY = currentMouseX, currentMouseY
            # rotate viewing point
            # get the x and y pixel coords
            if (diffMouseX == 0 and diffMouseY == 0):
                return

            # 180 degrees
            self.m_Axes.elev = art3d.norm_angle(self.m_Axes.elev - (diffMouseY/self.m_YAxisLength)*180)
            self.m_Axes.azim = art3d.norm_angle(self.m_Axes.azim - (diffMouseX/self.m_XAxisLength)*180)

            self.redrawAxes()

        # panning
        elif (cPlotConstants.m_MousePanButton == iEvent.button):
            diffMouseX, diffMouseY, diffMouseZ = [(n1 - n2)*cPlot3DConstants.m_PanSensitivity
                for (n1, n2) in zip (self.getCoord(currentMouseX, currentMouseY),
                    self.getCoord(self.m_PreviousMouseX, self.m_PreviousMouseY))]

            self.updateAxesData()

            if (False == self.m_LockAxes[eAxes.xAxis]):
                self.shiftXAxis(diffMouseX)

            if (False == self.m_LockAxes[eAxes.yAxis]):
                self.shiftYAxis(diffMouseY)

            if (False == self.m_LockAxes[eAxes.zAxis]):
                self.shiftZAxis(diffMouseZ)

            self.redrawAxes()


    def shiftZAxis(self, iShiftAmount):
        self.m_Axes.set_zlim([self.m_ZAxisMin + iShiftAmount, self.m_ZAxisMax + iShiftAmount])
        self.redrawAxes()


    def zoomAxes(self, iZoomAmount):
        self.updateAxesData()

        diffX = (self.m_XAxisMax - self.m_XAxisMin) * (0.1 * iZoomAmount)
        diffY = (self.m_YAxisMax - self.m_YAxisMin) * (0.1 * iZoomAmount)
        diffZ = (self.m_ZAxisMax - self.m_ZAxisMin) * (0.1 * iZoomAmount)

        self.m_Axes.set_xlim3d(self.m_XAxisMin + diffX, self.m_XAxisMax - diffX)
        self.m_Axes.set_ylim3d(self.m_YAxisMin + diffY, self.m_YAxisMax - diffY)
        self.m_Axes.set_zlim3d(self.m_ZAxisMin + diffZ, self.m_ZAxisMax - diffZ)

        self.redrawAxes()


    # Tries to get a 3D coordinate from a 2D screen
    # This function does not work very well unless it is zoomed into a considerable degree
    # modified from mpl.toolkits.mplot3d.axes3d.format_coord
    def getCoord(self, iPosX, iPosY):
        point = (iPosX, iPosY)
        edges = self.m_Axes.tunit_edges()

        #lines = [proj3d.line2d(p0,p1) for (p0,p1) in edges]
        ldists = [(proj3d.line2d_seg_dist(p0, p1, point), i) for i, (p0, p1) in enumerate(edges)]
        ldists.sort()
        # nearest edge
        edgei = ldists[0][1]

        p0, p1 = edges[edgei]

        # scale the z value to match
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        d0 = NP.hypot(x0-iPosX, y0-iPosY)
        d1 = NP.hypot(x1-iPosX, y1-iPosY)
        dt = d0+d1
        z = d1/dt * z0 + d0/dt * z1

        x, y, z = proj3d.inv_transform(iPosX, iPosY, z, self.m_Axes.M)
        return (x, y, z)


    def rescaleAxes(self):
        self.m_Axes.auto_scale_xyz(self.m_XData, self.m_YData, self.m_ZData, had_data=True)
