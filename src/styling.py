from tkinter import ttk

BACKGROUND_COLOR_1 = '#282828'
BACKGROUND_COLOR_2 = '#4d4d4d'
FONT_COLOR_1 = '#dddddd'
FONT_1 = ('TkDefaultFont', 11)


def create_auto_style_widget_class(widgetClass, styleNameModifier=''):
    class AutoStyleWidget(widgetClass):
        def __init__(self, master, **kwargs):
            super().__init__(master, **kwargs)
            self._style = AutoStyle()
            masterWidgetClass = self.master.winfo_class()
            selfWidgetClass = self.winfo_class()
            masterStyle = self.master.cget('style')
            styleClass = type(self).__name__ + '.' + masterStyle
            if selfWidgetClass != masterWidgetClass:
                styleClass = styleClass.replace('.', '_') + '.' + styleNameModifier + selfWidgetClass
                self._style.link_styles(masterStyle, styleClass)
            self.configure(style=styleClass)

    AutoStyleWidget.__name__ = widgetClass.__name__ + 'AutoStyle'
    AutoStyleWidget.__qualname__ = AutoStyleWidget.__name__
    return AutoStyleWidget


class AutoStyle(ttk.Style):
    __linkedStyles = {}
    __configuredStyles = {}

    def __init__(self):
        super().__init__()
        self._autoOptions = ('background',
                             'foreground',
                             'font')

    def configure(self, style, query_opt=None, **kwargs):
        super().configure(style, query_opt, **kwargs)
        self._set_child_styles_to_parent(style)
        if style in AutoStyle.__configuredStyles:
            for key in kwargs:
                if key in self._autoOptions:
                    AutoStyle.__configuredStyles[style].add(key)
        else:
            AutoStyle.__configuredStyles[style] = set()
            for key in kwargs:
                if key in self._autoOptions:
                    AutoStyle.__configuredStyles[style].add(key)

    def link_styles(self, parentStyle, childStyle):
        if parentStyle in AutoStyle.__linkedStyles:
            AutoStyle.__linkedStyles[parentStyle].add(childStyle)
        else:
            AutoStyle.__linkedStyles[parentStyle] = {childStyle}
        self._set_child_styles_to_parent(parentStyle)

    def configure_font_size(self, style, fontSize):
        # consider merging this method with self.configure
        font = self.lookup(style, 'font')
        font = font.split()
        font[-1] = fontSize
        font = tuple(font)
        self.configure(style, font=font)

    def _set_child_styles_to_parent(self, parentStyle):
        parentOptions = {}
        for option in self._autoOptions:
            parentOptions[option] = self.lookup(parentStyle, option)
        for childStyle in AutoStyle.__linkedStyles.get(parentStyle, []):
            kwargs = dict(parentOptions)
            for configuredOption in AutoStyle.__configuredStyles.get(childStyle, []):
                kwargs.pop(configuredOption)
            super().configure(childStyle, **kwargs)


FrameAutoStyle = create_auto_style_widget_class(ttk.Frame)
ScaleAutoStyle = create_auto_style_widget_class(ttk.Scale, 'Horizontal.')
LabelAutoStyle = create_auto_style_widget_class(ttk.Label)
CheckbuttonAutoStyle = create_auto_style_widget_class(ttk.Checkbutton)
