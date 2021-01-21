import sys
sys.path.append("../")
from cConstants import cOptionSimConstants, cBayesactSimConstants, cSliderConstants, cEPAConstants, cSystemConstants
from cEnum import eEPA
import threading
import wx
import cPlotEPA2D
from cPlotBayesactThread import cPlotBayesactThread
from cNumericValidatorTextBox import cNumericValidatorTextBox


class cBayesactSimGuiPanel(wx.Panel):
    # The parent here is the cGuiTabs, which holds the gui itself and the options too
    def __init__(self, parent, iBayesactSim, iOptionsBayesactPanel, iBayesactInteractiveGuiPanel=None, **kwargs):
        wx.Panel.__init__(self, parent, **kwargs)

        self.m_OptionsBayesactSimPanel = iOptionsBayesactPanel
        self.m_BayesactSim = iBayesactSim
        self.m_BayesactInteractiveGuiPanel = iBayesactInteractiveGuiPanel

        # You can use self.GetParent() to get the parent, but I like things explicit
        self.m_SimInteractiveTabs = parent



        # These are for all the options you can fill into the simulation
        ########################################################################################
        #self.m_SliderSize = (180, 24)
        #self.m_TextBoxSize = (110, 28)
        #self.m_ComboBoxSize = (110, 28)
        #self.m_ButtonSize = (190, 28)
        self.m_SliderSize = wx.DefaultSize
        self.m_TextBoxSize = wx.DefaultSize
        self.m_ComboBoxSize = wx.DefaultSize
        self.m_ButtonSize = (110, 30)

        self.m_StaticTextXAlign = 10
        self.m_TextboxXAlign = 170
        self.m_SliderXAlign = 290

        # This here specifies where the client and agent is aligned on the y co-ordinates
        # The reason why the initialization statements here look long is because I coded them to be relative to each other
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.m_ClientYAlign = 40
        self.m_AgentYAlign = 200
        self.m_HelpButtonXAlign = 60
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        macButtonYDiscrepancy = 0
        if (cSystemConstants.m_OS == cSystemConstants.m_MacOS):
            macButtonYDiscrepancy = -5

        # To draw a rectangle to outline the clients and agents
        self.Bind(wx.EVT_PAINT, self.onDrawOutlines)

        self.m_ClientStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_ClientText, pos=(self.m_StaticTextXAlign, self.m_ClientYAlign - 35))
        # Client Options
        #################################################################
        # The Client and Agent alpha and beta values
        # Box sizes for windows and mac are: (110 by 28) and (106 by 20) pixels respectively
        self.m_ClientAlphaStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_ClientAlpha, pos=(self.m_StaticTextXAlign, self.m_ClientYAlign))
        self.m_ClientAlphaTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_ClientAlphaStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize,
                                                value=str(cOptionSimConstants.m_ClientAlphaDefault),
                                                validator=cNumericValidatorTextBox(iDecimals=True, iNegative=False))
        self.m_ClientAlphaTextBox.Bind(wx.EVT_TEXT, self.onSetClientAlphaFlag)

        self.m_ClientAlphaButton = wx.Button(self, -1, "?", pos=(self.m_HelpButtonXAlign, self.m_ClientAlphaStaticText.GetPosition()[1]+macButtonYDiscrepancy), size=(20, 20))
        self.m_ClientAlphaButton.Bind(wx.EVT_BUTTON, lambda event, message="Alpha", title="Help": self.onSpawnMessageBox(event, message, title))

        # Default size on windows is (100, 24)
        self.m_ClientAlphaSlider = wx.Slider(self, -1, value=cOptionSimConstants.m_ClientAlphaDefault*cSliderConstants.m_Precision,
                                             pos=(self.m_SliderXAlign, self.m_ClientAlphaTextBox.GetPosition()[1]), size=self.m_SliderSize,
                                             minValue=cOptionSimConstants.m_MinAlpha*cSliderConstants.m_Precision,
                                             maxValue=cOptionSimConstants.m_MaxAlpha*cSliderConstants.m_Precision,
                                             style=wx.SL_HORIZONTAL)
        self.m_ClientAlphaSlider.Bind(wx.EVT_SCROLL, lambda event, textBox=self.m_ClientAlphaTextBox: self.onChangeValueViaSlider(event, textBox))


        self.m_ClientBetaOfClientStaticTextPosition = self.m_StaticTextXAlign, self.m_ClientAlphaStaticText.GetPosition()[1]+30
        self.m_ClientBetaOfClientStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_ClientBetaOfClient, pos=self.m_ClientBetaOfClientStaticTextPosition)
        self.m_ClientBetaOfClientSubscript_c_StaticText = wx.StaticText(self, -1, cOptionSimConstants.m_Subscript_c, pos=(self.m_ClientBetaOfClientStaticTextPosition[0]+8, self.m_ClientBetaOfClientStaticTextPosition[1]+5))

        self.m_ClientBetaOfClientTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_ClientBetaOfClientStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize,
                                                       value=str(cOptionSimConstants.m_ClientBetaOfClientDefault),
                                                       validator=cNumericValidatorTextBox(iDecimals=True, iNegative=False))
        self.m_ClientBetaOfClientTextBox.Bind(wx.EVT_TEXT, self.onSetClientBetaOfClientFlag)

        self.m_ClientBetaOfClientButton = wx.Button(self, -1, "?", pos=(self.m_HelpButtonXAlign, self.m_ClientBetaOfClientStaticText.GetPosition()[1]+macButtonYDiscrepancy), size=(20, 20))
        self.m_ClientBetaOfClientButton.Bind(wx.EVT_BUTTON, lambda event, message="Client's Beta of Client", title="Help": self.onSpawnMessageBox(event, message, title))

        self.m_ClientBetaOfClientSlider = wx.Slider(self, -1, value=cOptionSimConstants.m_ClientBetaOfClientDefault*cSliderConstants.m_Precision,
                                                    pos=(self.m_SliderXAlign, self.m_ClientBetaOfClientTextBox.GetPosition()[1]), size=self.m_SliderSize,
                                                    minValue=cOptionSimConstants.m_MinBeta*cSliderConstants.m_Precision,
                                                    maxValue=cOptionSimConstants.m_MaxBeta*cSliderConstants.m_Precision,
                                                    style=wx.SL_HORIZONTAL)
        self.m_ClientBetaOfClientSlider.Bind(wx.EVT_SCROLL, lambda event, textBox=self.m_ClientBetaOfClientTextBox: self.onChangeValueViaSlider(event, textBox))


        self.m_ClientBetaOfAgentStaticTextPosition = self.m_StaticTextXAlign, self.m_ClientBetaOfClientStaticText.GetPosition()[1]+30
        self.m_ClientBetaOfAgentStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_ClientBetaOfClient, pos=self.m_ClientBetaOfAgentStaticTextPosition)
        self.m_ClientBetaOfAgentSubscript_a_StaticText = wx.StaticText(self, -1, cOptionSimConstants.m_Subscript_a, pos=(self.m_ClientBetaOfAgentStaticTextPosition[0]+8, self.m_ClientBetaOfAgentStaticTextPosition[1]+5))

        self.m_ClientBetaOfAgentTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_ClientBetaOfAgentStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize,
                                                      value=str(cOptionSimConstants.m_ClientBetaOfAgentDefault),
                                                      validator=cNumericValidatorTextBox(iDecimals=True, iNegative=False))
        self.m_ClientBetaOfAgentTextBox.Bind(wx.EVT_TEXT, self.onSetClientBetaOfAgentFlag)

        self.m_ClientBetaOfAgentButton = wx.Button(self, -1, "?", pos=(self.m_HelpButtonXAlign, self.m_ClientBetaOfAgentStaticText.GetPosition()[1]+macButtonYDiscrepancy), size=(20, 20))
        self.m_ClientBetaOfAgentButton.Bind(wx.EVT_BUTTON, lambda event, message="Client's Beta of Agent", title="Help": self.onSpawnMessageBox(event, message, title))

        self.m_ClientBetaOfAgentSlider = wx.Slider(self, -1, value=cOptionSimConstants.m_ClientBetaOfAgentDefault*cSliderConstants.m_Precision,
                                                   pos=(self.m_SliderXAlign, self.m_ClientBetaOfAgentTextBox.GetPosition()[1]), size=self.m_SliderSize,
                                                   minValue=cOptionSimConstants.m_MinBeta*cSliderConstants.m_Precision,
                                                   maxValue=cOptionSimConstants.m_MaxBeta*cSliderConstants.m_Precision,
                                                   style=wx.SL_HORIZONTAL)
        self.m_ClientBetaOfAgentSlider.Bind(wx.EVT_SCROLL, lambda event, textBox=self.m_ClientBetaOfAgentTextBox: self.onChangeValueViaSlider(event, textBox))


        self.m_ClientGammaStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_ClientGamma, pos=(self.m_StaticTextXAlign, self.m_ClientBetaOfAgentStaticText.GetPosition()[1]+30))
        self.m_ClientGammaTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_ClientGammaStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize,
                                               value=str(cOptionSimConstants.m_GammaValueDefault),
                                               validator=cNumericValidatorTextBox(iDecimals=True, iNegative=False))
        self.m_ClientGammaTextBox.Bind(wx.EVT_TEXT, self.onSetClientGammaFlag)

        self.m_ClientGammaButton = wx.Button(self, -1, "?", pos=(self.m_HelpButtonXAlign, self.m_ClientGammaStaticText.GetPosition()[1]+macButtonYDiscrepancy), size=(20, 20))
        self.m_ClientGammaButton.Bind(wx.EVT_BUTTON, lambda event, message="Client's Gamma Value", title="Help": self.onSpawnMessageBox(event, message, title))

        self.m_ClientGammaSlider = wx.Slider(self, -1, value=cOptionSimConstants.m_GammaValueDefault*cSliderConstants.m_Precision,
                                            pos=(self.m_SliderXAlign, self.m_ClientGammaTextBox.GetPosition()[1]), size=self.m_SliderSize,
                                            minValue=cOptionSimConstants.m_MinGamma*cSliderConstants.m_Precision,
                                            maxValue=cOptionSimConstants.m_MaxGamma*cSliderConstants.m_Precision,
                                            style=wx.SL_HORIZONTAL)
        self.m_ClientGammaSlider.Bind(wx.EVT_SCROLL, lambda event, textBox=self.m_ClientGammaTextBox: self.onChangeValueViaSlider(event, textBox))


        #################################################################


        self.m_AgentStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_AgentText, pos=(self.m_StaticTextXAlign, self.m_AgentYAlign - 35))
        # Agent Options
        #################################################################
        self.m_AgentAlphaStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_AgentAlpha, pos=(self.m_StaticTextXAlign, self.m_AgentYAlign))
        self.m_AgentAlphaTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_AgentAlphaStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize,
                                               value=str(cOptionSimConstants.m_AgentAlphaDefault),
                                               validator=cNumericValidatorTextBox(iDecimals=True, iNegative=False))
        self.m_AgentAlphaTextBox.Bind(wx.EVT_TEXT, self.onSetAgentAlphaFlag)

        self.m_AgentAlphaButton = wx.Button(self, -1, "?", pos=(self.m_HelpButtonXAlign, self.m_AgentAlphaStaticText.GetPosition()[1]+macButtonYDiscrepancy), size=(20, 20))
        self.m_AgentAlphaButton.Bind(wx.EVT_BUTTON, lambda event, message="Alpha", title="Help": self.onSpawnMessageBox(event, message, title))

        self.m_AgentAlphaSlider = wx.Slider(self, -1, value=cOptionSimConstants.m_AgentAlphaDefault*cSliderConstants.m_Precision,
                                            pos=(self.m_SliderXAlign, self.m_AgentAlphaTextBox.GetPosition()[1]), size=self.m_SliderSize,
                                            minValue=cOptionSimConstants.m_MinAlpha*cSliderConstants.m_Precision,
                                            maxValue=cOptionSimConstants.m_MaxAlpha*cSliderConstants.m_Precision,
                                            style=wx.SL_HORIZONTAL)
        self.m_AgentAlphaSlider.Bind(wx.EVT_SCROLL, lambda event, textBox=self.m_AgentAlphaTextBox: self.onChangeValueViaSlider(event, textBox))


        self.m_AgentBetaOfClientStaticTextPosition = self.m_StaticTextXAlign, self.m_AgentAlphaStaticText.GetPosition()[1]+30
        self.m_AgentBetaOfClientStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_AgentBetaOfClient, pos=self.m_AgentBetaOfClientStaticTextPosition)
        self.m_AgentBetaOfClientSubscript_c_StaticText = wx.StaticText(self, -1, cOptionSimConstants.m_Subscript_c, pos=(self.m_AgentBetaOfClientStaticTextPosition[0]+8, self.m_AgentBetaOfClientStaticTextPosition[1]+5))

        self.m_AgentBetaOfClientTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_AgentBetaOfClientStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize,
                                                      value=str(cOptionSimConstants.m_AgentBetaOfClientDefault),
                                                      validator=cNumericValidatorTextBox(iDecimals=True, iNegative=False))
        self.m_AgentBetaOfClientTextBox.Bind(wx.EVT_TEXT, self.onSetAgentBetaOfClientFlag)

        self.m_AgentBetaOfClientButton = wx.Button(self, -1, "?", pos=(self.m_HelpButtonXAlign, self.m_AgentBetaOfClientStaticText.GetPosition()[1]+macButtonYDiscrepancy), size=(20, 20))
        self.m_AgentBetaOfClientButton.Bind(wx.EVT_BUTTON, lambda event, message="Agent's Beta of Client", title="Help": self.onSpawnMessageBox(event, message, title))

        self.m_AgentBetaOfClientSlider = wx.Slider(self, -1, value=cOptionSimConstants.m_AgentBetaOfClientDefault*cSliderConstants.m_Precision,
                                                   pos=(self.m_SliderXAlign, self.m_AgentBetaOfClientTextBox.GetPosition()[1]), size=self.m_SliderSize,
                                                   minValue=cOptionSimConstants.m_MinBeta*cSliderConstants.m_Precision,
                                                   maxValue=cOptionSimConstants.m_MaxBeta*cSliderConstants.m_Precision,
                                                   style=wx.SL_HORIZONTAL)
        self.m_AgentBetaOfClientSlider.Bind(wx.EVT_SCROLL, lambda event, textBox=self.m_AgentBetaOfClientTextBox: self.onChangeValueViaSlider(event, textBox))


        self.m_AgentBetaOfAgentStaticTextPosition = self.m_StaticTextXAlign, self.m_AgentBetaOfClientStaticText.GetPosition()[1]+30
        self.m_AgentBetaOfAgentStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_AgentBetaOfClient, pos=self.m_AgentBetaOfAgentStaticTextPosition)
        self.m_AgentBetaOfAgentSubscript_a_StaticText = wx.StaticText(self, -1, cOptionSimConstants.m_Subscript_a, pos=(self.m_AgentBetaOfAgentStaticTextPosition[0]+8, self.m_AgentBetaOfAgentStaticTextPosition[1]+5))

        self.m_AgentBetaOfAgentTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_AgentBetaOfAgentStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize,
                                                     value=str(cOptionSimConstants.m_AgentBetaOfAgentDefault),
                                                     validator=cNumericValidatorTextBox(iDecimals=True, iNegative=False))
        self.m_AgentBetaOfAgentTextBox.Bind(wx.EVT_TEXT, self.onSetAgentBetaOfAgentFlag)

        self.m_AgentBetaOfAgentButton = wx.Button(self, -1, "?", pos=(self.m_HelpButtonXAlign, self.m_AgentBetaOfAgentStaticText.GetPosition()[1]+macButtonYDiscrepancy), size=(20, 20))
        self.m_AgentBetaOfAgentButton.Bind(wx.EVT_BUTTON, lambda event, message="Agent's Beta of Agent", title="Help": self.onSpawnMessageBox(event, message, title))

        self.m_AgentBetaOfAgentSlider = wx.Slider(self, -1, value=cOptionSimConstants.m_AgentBetaOfAgentDefault*cSliderConstants.m_Precision,
                                                  pos=(self.m_SliderXAlign, self.m_AgentBetaOfAgentTextBox.GetPosition()[1]), size=self.m_SliderSize,
                                                  minValue=cOptionSimConstants.m_MinBeta*cSliderConstants.m_Precision,
                                                  maxValue=cOptionSimConstants.m_MaxBeta*cSliderConstants.m_Precision,
                                                 style=wx.SL_HORIZONTAL)
        self.m_AgentBetaOfAgentSlider.Bind(wx.EVT_SCROLL, lambda event, textBox=self.m_AgentBetaOfAgentTextBox: self.onChangeValueViaSlider(event, textBox))


        self.m_AgentGammaStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_AgentGamma, pos=(self.m_StaticTextXAlign, self.m_AgentBetaOfAgentStaticText.GetPosition()[1]+30))
        self.m_AgentGammaTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_AgentGammaStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize,
                                               value=str(cOptionSimConstants.m_GammaValueDefault),
                                               validator=cNumericValidatorTextBox(iDecimals=True, iNegative=False))
        self.m_AgentGammaTextBox.Bind(wx.EVT_TEXT, self.onSetAgentGammaFlag)

        self.m_AgentGammaButton = wx.Button(self, -1, "?", pos=(self.m_HelpButtonXAlign, self.m_AgentGammaStaticText.GetPosition()[1]+macButtonYDiscrepancy), size=(20, 20))
        self.m_AgentGammaButton.Bind(wx.EVT_BUTTON, lambda event, message="Agent's Gamma Value", title="Help": self.onSpawnMessageBox(event, message, title))

        self.m_AgentGammaSlider = wx.Slider(self, -1, value=cOptionSimConstants.m_GammaValueDefault*cSliderConstants.m_Precision,
                                            pos=(self.m_SliderXAlign, self.m_AgentGammaTextBox.GetPosition()[1]), size=self.m_SliderSize,
                                            minValue=cOptionSimConstants.m_MinGamma*cSliderConstants.m_Precision,
                                            maxValue=cOptionSimConstants.m_MaxGamma*cSliderConstants.m_Precision,
                                            style=wx.SL_HORIZONTAL)
        self.m_AgentGammaSlider.Bind(wx.EVT_SCROLL, lambda event, textBox=self.m_AgentGammaTextBox: self.onChangeValueViaSlider(event, textBox))


        #################################################################


        self.m_EnvironmentNoiseStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_EnvironmentNoise, pos=(self.m_StaticTextXAlign, 340))
        self.m_EnvironmentNoiseTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_EnvironmentNoiseStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize,
                                                     value=str(cOptionSimConstants.m_EnvironmentNoiseDefault),
                                                     validator=cNumericValidatorTextBox(iDecimals=True, iNegative=False))
        self.m_EnvironmentNoiseTextBox.Bind(wx.EVT_TEXT, self.onSetEnvironmentNoiseFlag)

        self.m_EnvironmentNoiseSlider = wx.Slider(self, -1, value=cOptionSimConstants.m_EnvironmentNoiseDefault*cSliderConstants.m_Precision,
                                                  pos=(self.m_SliderXAlign, self.m_EnvironmentNoiseTextBox.GetPosition()[1]), size=self.m_SliderSize,
                                                  minValue=cOptionSimConstants.m_MinEnvironmentNoise*cSliderConstants.m_Precision,
                                                  maxValue=cOptionSimConstants.m_MaxEnvironmentNoise*cSliderConstants.m_Precision,
                                                  style=wx.SL_HORIZONTAL)
        self.m_EnvironmentNoiseSlider.Bind(wx.EVT_SCROLL, lambda event, textBox=self.m_EnvironmentNoiseTextBox: self.onChangeValueViaSlider(event, textBox))



        self.m_NumberOfStepsStaticText = wx.StaticText(self, -1, cOptionSimConstants.m_NumSteps, pos=(self.m_StaticTextXAlign, 370))
        self.m_NumberOfStepsTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_NumberOfStepsStaticText.GetPosition()[1] - 2), size=self.m_TextBoxSize,
                                                  value=str(cOptionSimConstants.m_NumStepsDefault),
                                                  validator=cNumericValidatorTextBox(iDecimals=False, iNegative=False))
        self.m_NumberOfStepsTextBox.Bind(wx.EVT_TEXT, self.onSetNumStepsFlag)

        self.m_CurrentIterationsStaticText = wx.StaticText(self, -1, "Current Iteration", pos=(self.m_StaticTextXAlign, 400))
        self.m_CurrentIterationsTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_CurrentIterationsStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize, style=wx.TE_READONLY)
        self.m_CurrentIterationsTextBox.SetValue(str(self.m_BayesactSim.total_iterations))

        self.m_PreviousIterationsStaticText = wx.StaticText(self, -1, "Previous Iteration", pos=(self.m_StaticTextXAlign, 430))
        self.m_PreviousIterationsTextBox = wx.TextCtrl(self, -1, pos=(self.m_TextboxXAlign, self.m_PreviousIterationsStaticText.GetPosition()[1]-2), size=self.m_TextBoxSize, style=wx.TE_READONLY)
        self.m_PreviousIterationsTextBox.SetValue("0")


        ########################################################################################


        self.m_StartButton = wx.Button(self, -1, label="Start", pos=(10, 470), size=self.m_ButtonSize)
        self.m_StartButton.Bind(wx.EVT_BUTTON, self.onStartBayesactSim)

        self.m_StepButton = wx.Button(self, -1, label="Step", pos=(140, 470), size=self.m_ButtonSize)
        self.m_StepButton.Bind(wx.EVT_BUTTON, self.onStepBayesactSim)

        self.m_StopButton = wx.Button(self, -1, label="Stop", pos=(270, 470), size=self.m_ButtonSize)
        self.m_StopButton.Bind(wx.EVT_BUTTON, self.onStopBayesactSim)

        #self.m_PauseButton = wx.Button(self, -1, label="Pause", pos=(10, 400), size=(190, 28))
        #self.m_PauseButton.Bind(wx.EVT_BUTTON, self.onPauseBayesactSim)

        #self.m_ResumeButton = wx.Button(self, -1, label="Resume", pos=(10, 430), size=(190, 28))
        #self.m_ResumeButton.Bind(wx.EVT_BUTTON, self.onResumeBayesactSim)

        #self.m_StopButton = wx.Button(self, -1, label="Stop", pos=(10, 460), size=(190, 28))
        #self.m_StopButton.Bind(wx.EVT_BUTTON, self.onStopBayesactSim)


        macImageYDiscrepancy = 0
        if (cSystemConstants.m_OS == cSystemConstants.m_MacOS):
            macImageYDiscrepancy = -3


        # The set mask colour (255, 255, 255) which is white, becomes transparent
        bmp = wx.Image("./gui/images/circleMagenta.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_GreenCircle = wx.StaticBitmap(self, -1, bmp, pos=(10, 525+macImageYDiscrepancy), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_Green2StaticText = wx.StaticText(self, -1, "What Client thinks of self", pos=(30, 520))

        bmp = wx.Image("./gui/images/circleRed.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_PinkCircle = wx.StaticBitmap(self, -1, bmp, pos=(10, 545+macImageYDiscrepancy), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_Pink2StaticText = wx.StaticText(self, -1, "What Client thinks of agent", pos=(30, 540))

        bmp = wx.Image("./gui/images/circleCyan.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_GoldenrodCircle = wx.StaticBitmap(self, -1, bmp, pos=(10, 565+macImageYDiscrepancy), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_Yellow2StaticText = wx.StaticText(self, -1, "What Agent thinks of self", pos=(30, 560))


        bmp = wx.Image("./gui/images/circleBlue.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_BlueCircle = wx.StaticBitmap(self, -1, bmp, pos=(10, 585+macImageYDiscrepancy), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_Blue2StaticText = wx.StaticText(self, -1, "What Agent thinks of client", pos=(30, 580))

        bmp = wx.Image("./gui/images/starMagenta.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_CyanStar = wx.StaticBitmap(self, -1, bmp, pos=(240, 523+macImageYDiscrepancy), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_CyanStarStaticText = wx.StaticText(self, -1, "Client action", pos=(260, 520))

        bmp = wx.Image("./gui/images/starCyan.png", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        bmp.SetMaskColour((255, 255, 255))
        self.m_YellowStar = wx.StaticBitmap(self, -1, bmp, pos=(240, 563+macImageYDiscrepancy), size=(bmp.GetWidth(), bmp.GetHeight()))
        self.m_YellowStarStaticText = wx.StaticText(self, -1, "Agent Action", pos=(260, 560))


        self.updateSettingsFromBayesact()



    # The reason I manually did this is because the supplied in wx.DC DrawRectangle has a fill
    def drawUnfilledRectangle(self, iDC, iX1, iY1, iX2, iY2, iColour=wx.BLACK, iBrushWidth=1):
        iDC.SetPen(wx.Pen(iColour, iBrushWidth))
        iDC.DrawLine(iX1, iY1, iX1, iY2)
        iDC.DrawLine(iX1, iY1, iX2, iY1)
        iDC.DrawLine(iX1, iY2, iX2, iY2)
        iDC.DrawLine(iX2, iY1, iX2, iY2)


    # Draws outlines and shapes, including circles
    def onDrawOutlines(self, iEvent=None):
        # stands for device context
        dc = wx.PaintDC(self)

        # The sliders are also much longer in the mac operating system, so we need to fix that
        rectangleXDiscrepancy = 0
        if (cSystemConstants.m_MacOS == cSystemConstants.m_OS):
            rectangleXDiscrepancy = 50

        # To draw the outline for the client options
        # 150 is from 4 textboxes * 30
        self.drawUnfilledRectangle(dc, 5, self.m_ClientYAlign - 10, 400 + rectangleXDiscrepancy, self.m_ClientYAlign + 120, wx.BLACK, 1)

        # To draw the outline for the agent options
        self.drawUnfilledRectangle(dc, 5, self.m_AgentYAlign - 10, 400 + rectangleXDiscrepancy, self.m_AgentYAlign + 120, wx.BLACK, 1)



    def onValueChange(self, iEvent):
        print iEvent.GetEventObject().GetValue()


    def onSpawnMessageBox(self, iEvent, iMessage, iTitle):
        wx.MessageBox(iMessage, iTitle)

    # To set the values of the gui to the values in bayesact
    def updateSettingsFromBayesact(self):
        self.m_OptionsBayesactSimPanel.updateSettingsFromBayesact()


    # To set the values of bayesact to the values in the gui
    # Should only be used to initialize
    def updateBayesactFromSettings(self):
        self.m_BayesactSim.client_alpha_value = float(self.m_ClientAlphaTextBox.GetValue())

        self.m_BayesactSim.client_beta_value_of_client = float(self.m_ClientBetaOfClientTextBox.GetValue())
        self.m_BayesactSim.client_beta_value_of_agent = float(self.m_ClientBetaOfAgentTextBox.GetValue())
        self.m_BayesactSim.client_gamma_value = float(self.m_ClientGammaTextBox.GetValue())

        self.m_BayesactSim.agent_alpha_value = float(self.m_AgentAlphaTextBox.GetValue())

        self.m_BayesactSim.agent_beta_value_of_client = float(self.m_AgentBetaOfClientTextBox.GetValue())
        self.m_BayesactSim.agent_beta_value_of_agent = float(self.m_AgentBetaOfAgentTextBox.GetValue())
        self.m_BayesactSim.agent_gamma_value = float(self.m_AgentGammaTextBox.GetValue())

        self.m_BayesactSim.env_noise = float(self.m_EnvironmentNoiseTextBox.GetValue())
        self.m_BayesactSim.gamma_value = float(self.m_AgentGammaTextBox.GetValue())

        self.m_OptionsBayesactSimPanel.updateBayesactFromSettings()


    def disableStartingOptions(self):
        self.m_OptionsBayesactSimPanel.disableStartingOptions()

    def enableStartingOptions(self):
        self.m_OptionsBayesactSimPanel.enableStartingOptions()

    def onSetClientAlphaFlag(self, iEvent):
        self.m_BayesactSim.update_client_alpha = True
        value = iEvent.GetEventObject().GetValue()
        self.onChangeSliderViaValue(self.m_ClientAlphaSlider, value)

    def onSetClientBetaOfClientFlag(self, iEvent):
        self.m_BayesactSim.update_client_beta_of_client = True
        value = iEvent.GetEventObject().GetValue()
        self.onChangeSliderViaValue(self.m_ClientBetaOfClientSlider, value)

    def onSetClientBetaOfAgentFlag(self, iEvent):
        self.m_BayesactSim.update_client_beta_of_agent = True
        value = iEvent.GetEventObject().GetValue()
        self.onChangeSliderViaValue(self.m_ClientBetaOfAgentSlider, value)

    def onSetClientGammaFlag(self, iEvent):
        self.m_BayesactSim.update_client_gamma = True
        value = iEvent.GetEventObject().GetValue()
        self.onChangeSliderViaValue(self.m_ClientGammaSlider, value)

    def onSetAgentAlphaFlag(self, iEvent):
        self.m_BayesactSim.update_agent_alpha = True
        value = iEvent.GetEventObject().GetValue()
        self.onChangeSliderViaValue(self.m_AgentAlphaSlider, value)

    def onSetAgentBetaOfClientFlag(self, iEvent):
        self.m_BayesactSim.update_agent_beta_value_of_client = True
        value = iEvent.GetEventObject().GetValue()
        self.onChangeSliderViaValue(self.m_AgentBetaOfClientSlider, value)

    def onSetAgentBetaOfAgentFlag(self, iEvent):
        self.m_BayesactSim.update_agent_beta_value_of_agent = True
        value = iEvent.GetEventObject().GetValue()
        self.onChangeSliderViaValue(self.m_AgentBetaOfAgentSlider, value)

    def onSetAgentGammaFlag(self, iEvent):
        self.m_BayesactSim.update_agent_gamma = True
        value = iEvent.GetEventObject().GetValue()
        self.onChangeSliderViaValue(self.m_AgentGammaSlider, value)

    def onSetEnvironmentNoiseFlag(self, iEvent):
        self.m_BayesactSim.update_environment_noise = True
        value = iEvent.GetEventObject().GetValue()
        self.onChangeSliderViaValue(self.m_EnvironmentNoiseSlider, value)


    def onSetNumStepsFlag(self, iEvent):
        self.m_BayesactSim.update_num_steps = True


    def onChangeSliderViaValue(self, iSlider, iValue):
        if ("" != iValue):
            iSlider.SetValue(float(iValue)*cSliderConstants.m_Precision)

    def onChangeValueViaSlider(self, iEvent, iTextBox):
        iTextBox.SetValue(str(iEvent.GetEventObject().GetValue() / cSliderConstants.m_Precision))


    def onStepBayesactSim(self, iEvent):
        if (None != self.m_SimInteractiveTabs.m_BayesactSimThread):
            self.m_BayesactSim.step_bayesact_sim = True
            self.m_BayesactSim.thread_event.set()
        else:
            self.m_BayesactSim.update_num_steps = False
            self.onStartBayesactSim(None)
            self.m_BayesactSim.initial_step_bayesact_sim = True
            self.m_BayesactSim.num_steps = int(self.m_NumberOfStepsTextBox.GetValue())



    # Stops bayesact sim and kills thread
    def onStopBayesactSim(self, iEvent=None):
        if (None != self.m_SimInteractiveTabs.m_BayesactSimThread):
            self.m_BayesactSim.terminate_flag = True
            self.m_BayesactSim.thread_event.set()
            # Thread automatically joins, no need to tell it to join again
            self.m_SimInteractiveTabs.m_BayesactSimThread = None
            self.enableStartingOptions()


    def onStartBayesactSim(self, iEvent):
        #self.m_StartButton.SetLabel("Restart")
        self.onStopBayesactSim()

        self.disableStartingOptions()

        self.updateBayesactFromSettings()

        self.m_BayesactSim.terminate_flag = False

        self.m_SimInteractiveTabs.m_BayesactSimThread = threading.Thread(target=self.m_BayesactSim.startThread)

        self.m_BayesactSim.thread_event = threading.Event()

        self.m_SimInteractiveTabs.m_Plotter.setThread(self.m_SimInteractiveTabs.m_BayesactSimThread)
        self.m_SimInteractiveTabs.m_Plotter.startThread()



