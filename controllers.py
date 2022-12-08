import views
import models
import threading
from math import exp, log
from scipy import special


class MasterController:
    def __init__(self):
        self._root = views.Root()
        self._tabControllers = []
        self._maxDisplayImageSize = None
        self._root.bind_exit_button(self.exit_button_callback)
        self._root.menuBar.bind_button(self._root.menuBar.ButtonType.OPEN, self.open_button_callback)
        self._root.menuBar.bind_button(self._root.menuBar.ButtonType.SAVE, self.save_button_callback)
        self._root.menuBar.bind_button(self._root.menuBar.ButtonType.SAVE_AS, self.save_as_button_callback)
        self._root.menuBar.bind_button(self._root.menuBar.ButtonType.CLOSE, self.close_button_callback)
        self._root.menuBar.bind_button(self._root.menuBar.ButtonType.EXIT, self.exit_button_callback)
        self._root.mainloop()

    def open_button_callback(self):
        filePath = views.open_file_dialog()
        if filePath:
            self._tabControllers.append(_TabController(filePath, self._root.fileTabs, self._maxDisplayImageSize))
            if self._maxDisplayImageSize is None:
                self._maxDisplayImageSize = self._tabControllers[0].maxDisplayImageSize
            self._root.switch_to_tab(len(self._tabControllers)-1)
            self._root.menuBar.enable_button(self._root.menuBar.ButtonType.SAVE)
            self._root.menuBar.enable_button(self._root.menuBar.ButtonType.SAVE_AS)
            self._root.menuBar.enable_button(self._root.menuBar.ButtonType.CLOSE)

    def save_as_button_callback(self):
        self._tabControllers[self._root.currentTab].save_as_button_callback()

    def save_button_callback(self):
        self._tabControllers[self._root.currentTab].save_button_callback()

    def close_button_callback(self):
        saveChanges = False
        if self._tabControllers[self._root.currentTab].existUnsavedChanges:
            saveChanges = views.unsaved_changes_dialog(self._tabControllers[self._root.currentTab].fileName)
        if saveChanges is not None:
            if saveChanges:
                self.save_button_callback()
            self._tabControllers.pop(self._root.currentTab)
            self._root.close_tab(self._root.currentTab)
            if not self._tabControllers:
                self._root.menuBar.disable_button(self._root.menuBar.ButtonType.SAVE)
                self._root.menuBar.disable_button(self._root.menuBar.ButtonType.SAVE_AS)
                self._root.menuBar.disable_button(self._root.menuBar.ButtonType.CLOSE)

    def exit_button_callback(self):
        while self._tabControllers:
            tabId = len(self._tabControllers)-1
            self._root.switch_to_tab(tabId)
            self.close_button_callback()
            if len(self._tabControllers)-1 == tabId:
                break
        if not self._tabControllers:
            self._root.destroy()


class _TabController:
    def __init__(self, filePath, tabContainer, maxDisplayImageSize):
        self._existUnsavedChanges = False
        self.fileName = filePath.split('/')[-1]
        self._tab = views.Tab(tabContainer, tabTitle=self.fileName)
        if maxDisplayImageSize is None:
            self.maxDisplayImageSize = self._tab.maxImageSize
        else:
            self.maxDisplayImageSize = maxDisplayImageSize
        self._model = models.Model(filePath, self.maxDisplayImageSize)
        self._tab.add_imageDisplay(self._model.processedDisplayImage)
        self._tab.adjustmentsPanel.equalizeSliderGroup.bind_callback(self.equalize_slider_callback)
        self._tab.adjustmentsPanel.brightnessSliderGroup.bind_callback(self.brightness_slider_callback)
        self._tab.adjustmentsPanel.contrastSliderGroup.bind_callback(self.contrast_slider_callback)
        self._tab.adjustmentsPanel.saturationSliderGroup.bind_callback(self.saturation_slider_callback)
        self._tab.adjustmentsPanel.warmthSliderGroup.bind_callback(self.warmth_slider_callback)
        self._tab.adjustmentsPanel.tintSliderGroup.bind_callback(self.tint_slider_callback)
        self._tab.adjustmentsPanel.twoToneHueSliderGroup.bind_callback(self.two_tone_hue_callback)
        self._tab.adjustmentsPanel.twoToneSaturationSliderGroup.bind_callback(self.two_tone_saturation_callback)
        self._tab.adjustmentsPanel.bind_checkbox(self.checkbox_callback)
        self._model.add_processedImage_callback(self.processed_image_callback)
        self._model.add_existUnsavedChanges_callback(self.unsaved_changes_callback)

    @property
    def existUnsavedChanges(self):
        return self._model.existUnsavedChanges

    def equalize_slider_callback(self, event):
        event = float(event)
        event = 2*(event-.5)
        t = 90*special.erfinv(event*0.9998817874182897)
        self._model.change_processing_params({models.ParamType.EQUALIZE: t})

    def brightness_slider_callback(self, event):
        event = float(event)
        brightness = 2**((event-.5)*4)
        self._model.change_processing_params({models.ParamType.BRIGHTNESS: brightness})

    def contrast_slider_callback(self, event):
        event = float(event)
        contrast = 2**((event-.5)*2)
        self._model.change_processing_params({models.ParamType.CONTRAST: contrast})

    def saturation_slider_callback(self, event):
        event = float(event)
        saturation = 1.5*exp((event-.5)*log(9))-.5
        self._model.change_processing_params({models.ParamType.SATURATION: saturation})

    def warmth_slider_callback(self, event):
        event = float(event)
        x = (event-.5)*2
        warmth = (x**3+x)/2
        self._model.change_processing_params({models.ParamType.WARMTH: warmth})

    def tint_slider_callback(self, event):
        event = float(event)
        x = (event-.5)*2
        tintFactor = (x**3+x)/2
        self._model.change_processing_params({models.ParamType.TINT: tintFactor})

    def two_tone_hue_callback(self, event):
        event = float(event)
        twoToneHue = (event-.5)*180
        self._model.change_processing_params({models.ParamType.TWO_TONE_HUE: twoToneHue})

    def two_tone_saturation_callback(self, event):
        event = float(event)
        twoToneSaturation = 1.5*exp((event-.5)*log(9))-.5
        self._model.change_processing_params({models.ParamType.TWO_TONE_SATURATION: twoToneSaturation})

    def checkbox_callback(self):
        if self._tab.adjustmentsPanel.checkboxChecked:
            displayImage = self._model.originalDisplayImage
        else:
            displayImage = self._model.processedDisplayImage
        self._tab.imageDisplay.update_image(displayImage)

    def processed_image_callback(self, displayImage):
        if not self._tab.adjustmentsPanel.checkboxChecked:
            self._tab.imageDisplay.update_image(displayImage)

    def unsaved_changes_callback(self, existUnsavedChanges):
        self._tab.update_tab_title_to_save_state(existUnsavedChanges)

    def save_as_button_callback(self):
        filePath = views.save_file_dialog(self._model.filePath.split('/')[-1])
        if filePath:
            thread = threading.Thread(target=self._model.save_image, args=(filePath,))
            thread.start()

    def save_button_callback(self):
        thread = threading.Thread(target=self._model.save_image, args=(self._model.filePath,))
        thread.start()
