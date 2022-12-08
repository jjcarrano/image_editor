import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import messagebox
import enum
import linkable_widgets as lw
import styling as stl


def open_file_dialog():
    filePath = filedialog.askopenfilename()
    return filePath


def save_file_dialog(initialName=''):
    filePath = filedialog.asksaveasfilename(defaultextension='.jpg',
                                            initialfile=initialName,
                                            filetypes=(('JPEG', '*.jpg'), ('PNG', '*.png')))
    return filePath


def unsaved_changes_dialog(fileName):
    response = messagebox.askyesnocancel('', 'Save changes to {} before closing?'.format(fileName))
    return response


class Root(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Image Processor')
        self.state('zoomed')
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)
        self.iconbitmap('icon.ico')
        self.menuBar = _MenuBar(self)
        self.config(menu=self.menuBar)
        style = stl.AutoStyle()
        style.layout('TNotebook', [])
        self.fileTabs = ttk.Notebook(self)
        self.fileTabs.pack(expand=True, fill=tk.BOTH)
        self.fileTabs.enable_traversal()
        self.bind_all('<Button-1>', lambda event: event.widget.focus_set())
        self.bind('<MouseWheel>', self._on_mouse_wheel)
        self.event_add('<<VirtualMouseWheel>>', '<MouseWheel>')

    @property
    def currentTab(self):
        return self.fileTabs.index("current")

    def close_tab(self, tabId):
        self.fileTabs.forget(tabId)

    def switch_to_tab(self, tabId):
        self.fileTabs.select(tabId)

    def bind_exit_button(self, command):
        self.protocol("WM_DELETE_WINDOW", command)

    def _on_mouse_wheel(self, event):
        widgetWithFocus = self.focus_get()
        widgetWithFocus.event_generate('<<VirtualMouseWheel>>', state=event.delta)


class _MenuBar(tk.Menu):
    def __init__(self, container):
        super().__init__(container)
        self._fileMenu = tk.Menu(self, tearoff=False)
        self._add_fileMenu_buttons()
        self.add_cascade(label='File', menu=self._fileMenu)
        self.disable_button(self.ButtonType.SAVE)
        self.disable_button(self.ButtonType.SAVE_AS)
        self.disable_button(self.ButtonType.CLOSE)
        for typeOfButton in self.ButtonType:
            self._key_bind(typeOfButton.value[1], typeOfButton)

    def disable_button(self, typeOfButton):
        self._fileMenu.entryconfigure(self._return_button_type_index(typeOfButton), state=tk.DISABLED)

    def enable_button(self, typeOfButton):
        self._fileMenu.entryconfigure(self._return_button_type_index(typeOfButton), state=tk.NORMAL)

    def bind_button(self, typeOfButton, command):
        self._fileMenu.entryconfigure(self._return_button_type_index(typeOfButton), command=command)

    def _return_button_type_index(self, typeOfButton):
        return list(self.ButtonType).index(typeOfButton)

    def _add_fileMenu_buttons(self):
        for typeOfButton in self.ButtonType:
            shortCut = typeOfButton.value[1]
            shortCut = shortCut.replace('Control', 'Ctrl')
            shortCut = shortCut.replace('-', '+')
            shortCut = shortCut[:-1]+shortCut[-1].upper()
            label = typeOfButton.value[0]+' '*5+shortCut
            self._fileMenu.add_command(label=label)

    def _key_bind(self, keySequence, typeOfButton):
        keySequenceList = [keySequence, keySequence[:-1]+keySequence[-1].swapcase()]
        buttonIndex = self._return_button_type_index(typeOfButton)
        for sequence in keySequenceList:
            self.master.bind('<{}>'.format(sequence), lambda event: self._fileMenu.invoke(buttonIndex))

    class ButtonType(enum.Enum):
        # first element in tuple is label that appears in file menu, second element is key sequence used for shortcut
        OPEN = ('Open...', 'Control-o')
        SAVE = ('Save', 'Control-s')
        SAVE_AS = ('Save As...', 'Control-Shift-s')
        CLOSE = ('Close', 'Control-w')
        EXIT = ('Exit', 'Control-q')


class Tab(stl.FrameAutoStyle):
    def __init__(self, container, tabTitle=''):

        def determine_max_image_size():
            # only accurate if tab is selected when called
            testFrame = stl.FrameAutoStyle(self)
            testFrame.pack(fill=tk.BOTH, expand=True)
            testFrame.update()
            frameHeight = testFrame.winfo_height()
            margin = round(frameHeight*.2)
            maxHeight = frameHeight-margin
            maxWidth = testFrame.winfo_width()-margin
            testFrame.destroy()
            return maxHeight, maxWidth

        super().__init__(container, style='TFrame')
        self._style.configure(self.cget('style'), background=stl.BACKGROUND_COLOR_1)
        self.imageDisplay = None
        self.adjustmentsPanel = _AdjustmentsPanel(self)
        self.adjustmentsPanel.pack(side=tk.RIGHT, fill=tk.Y)
        self._tabTitle = tabTitle
        container.add(self, text=tabTitle)
        self.maxImageSize = determine_max_image_size()  # only accurate if tab is selected when called

    def add_imageDisplay(self, displayImage):
        if self.imageDisplay is None:
            self.imageDisplay = _ImageDisplay(self, displayImage)
            self.imageDisplay.pack(side=tk.LEFT, expand=True)

    def update_tab_title_to_save_state(self, existUnsavedChanges):
        if existUnsavedChanges:
            tabTitle = self._tabTitle + '*'
        else:
            tabTitle = self._tabTitle
        self.master.tab(self, text=tabTitle)


class _ImageDisplay(stl.FrameAutoStyle):
    def __init__(self, container, displayImage):
        super().__init__(container)
        self._displayImage = ImageTk.PhotoImage(image=Image.fromarray(displayImage))
        self._canvas = tk.Canvas(self,
                                 height=displayImage.shape[0],
                                 width=displayImage.shape[1],
                                 bd=0,
                                 highlightthickness=0)
        self._canvas.create_image(0, 0, anchor='nw', image=self._displayImage)
        self._canvas.pack()

    def update_image(self, displayImage):
        self._displayImage.paste(Image.fromarray(displayImage))


class _AdjustmentsPanel(stl.FrameAutoStyle):
    def __init__(self, container):
        super().__init__(container)
        self._style.configure(self.cget('style'),
                              background=stl.BACKGROUND_COLOR_2,
                              foreground=stl.FONT_COLOR_1,
                              font=stl.FONT_1)
        self._frame = stl.FrameAutoStyle(self)
        self.equalizeSliderGroup = _SliderGroup(self._frame, title='Equalize')
        self.brightnessSliderGroup = _SliderGroup(self._frame, title='Brightness')
        self.contrastSliderGroup = _SliderGroup(self._frame, title='Contrast')
        self.saturationSliderGroup = _SliderGroup(self._frame, title='Saturation')
        self.warmthSliderGroup = _SliderGroup(self._frame, title='Warmth')
        self.tintSliderGroup = _SliderGroup(self._frame, title='Tint')
        self.twoToneHueSliderGroup = _SliderGroup(self._frame, title='Two Tone Hue', valueRange=(-90, 90))
        self.twoToneSaturationSliderGroup = _SliderGroup(self._frame, title='Two Tone Saturation')
        self._checkboxFlag = tk.IntVar(0)
        self.beforeAfterCheckbox = stl.CheckbuttonAutoStyle(self._frame,
                                                            variable=self._checkboxFlag,
                                                            text='Toggle Before/After',
                                                            takefocus=False)
        self._style.configure_font_size(self.beforeAfterCheckbox.cget('style'), fontSize=10)
        self._frame.pack(expand=True, padx=7)
        sliderPadding = (0, 10)
        self.equalizeSliderGroup.pack(pady=sliderPadding)
        self.brightnessSliderGroup.pack(pady=sliderPadding)
        self.contrastSliderGroup.pack(pady=sliderPadding)
        self.saturationSliderGroup.pack(pady=sliderPadding)
        self.warmthSliderGroup.pack(pady=sliderPadding)
        self.tintSliderGroup.pack(pady=sliderPadding)
        self.twoToneHueSliderGroup.pack(pady=sliderPadding)
        self.twoToneSaturationSliderGroup.pack()
        self.beforeAfterCheckbox.pack(fill=tk.X, pady=(14, 0))
        self.beforeAfterCheckbox.bind('<FocusIn>', self._on_checkbox_focus)

    @property
    def checkboxChecked(self):
        return bool(self._checkboxFlag.get())

    def bind_checkbox(self, command):
        self.beforeAfterCheckbox.config(command=command)

    def _on_checkbox_focus(self, event):
        # prevents dashed outline from appearing around checkbox when clicked
        self.focus_set()


class _SliderGroup(stl.FrameAutoStyle):
    def __init__(self, container, title, valueRange=(-100, 100), initialValue=0):
        super().__init__(container)
        self._command = lambda *args, **kwargs: None
        self._title = stl.LabelAutoStyle(self, text=title, anchor='w')
        self._slider = lw.LinkableSlider(self, value=.5, length=240, takefocus=False)
        self._entryBox = lw.LinkableEntry(self, valueRange, initialValue,
                                          width=4,
                                          bg='#454545',
                                          fg=stl.FONT_COLOR_1,
                                          insertbackground=stl.FONT_COLOR_1,
                                          takefocus=False)
        lw.link_widgets(self._slider, self._entryBox, lw.range_converter)
        self._slider.pack(side=tk.BOTTOM, pady=(3, 0))
        self._title.pack(side=tk.LEFT, fill=tk.X)
        self._entryBox.pack(side=tk.RIGHT)
        self._slider.bind_external_command(self.slider_callback)

    def slider_callback(self, event):
        self._entryBox.focus_set()
        self._command(event)

    def bind_callback(self, command):
        self._command = command
