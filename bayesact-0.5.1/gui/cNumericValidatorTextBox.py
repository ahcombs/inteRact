import string
import unicodedata
import wx

class cNumericValidatorTextBox(wx.PyValidator):
    def __init__(self, iDecimals=True, iNegative=True):
        wx.PyValidator.__init__(self)

        self.m_AllowDecimals = iDecimals
        self.m_AllowNegatives = iNegative

        self.Bind(wx.EVT_CHAR, self.onChar)

    # This is a necessary function for wx.PyValidator's pure virtual function
    def Clone(self):
        return cNumericValidatorTextBox(iDecimals=self.m_AllowDecimals, iNegative=self.m_AllowNegatives)

    def Validate(self, win):
        textctrl = self.GetWindow()
        value = textctrl.GetValue()

        if ("" == value):
            return True

        value = isNumeric(value)
        
        if (value):
            if ((0 > value) and (not(self.m_AllowNegatives))):
                return False
            elif (0 != (value % 1)) and (not(self.m_AllowDecimals)):
                return False
            else:
                return True


    def onChar(self, iEvent):
        key = iEvent.GetKeyCode()
        value = self.GetWindow().GetValue()
        cursorIndex = self.GetWindow().GetInsertionPoint()

        # Allow spaces and backspaces
        if key < wx.WXK_SPACE or key == wx.WXK_DELETE or key > 255:
            iEvent.Skip()
            return

        # Allow numbers
        if (chr(key) in string.digits):
            iEvent.Skip()
            return
        # Allow hyphen for negative numbers if cursor is at beginning and no other hyphens exists
        if (self.m_AllowNegatives and (chr(key) == "-") and (0 == cursorIndex) and (not("-" in value))):
            iEvent.Skip()
            return
        # Allow period for decimals if no other periods events
        elif (self.m_AllowDecimals and (chr(key) == ".") and (not("." in value))):
            iEvent.Skip()
            return
        return

    def isNumeric(iString):
        try:
            return float(iString)
        except ValueError:
            pass

        try:
            return unicodedata.numeric(iString)
        except (TypeError, ValueError):
            pass

        return False