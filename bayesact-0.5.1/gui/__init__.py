import wx

class cMenuBar():
	def __init__(self):
		menuBar = wx.MenuBar()
		fileMenu = wx.Menu()
		fileMenu.Append(wx.ID_ANY, "Save as PNG", "")