import tkinter as tk
from abc import ABC, abstractmethod
import styling as stl


def link_widgets(widget1, widget2, outgoingConversionFunction):
    # consider making binding of external command part of this function
    widget1.linkCommand = widget2.set_value
    widget1.outgoingConverter = _OutgoingConverter(widget1.range, widget2.range, outgoingConversionFunction)
    widget2.linkCommand = widget1.set_value
    widget2.outgoingConverter = _OutgoingConverter(widget2.range, widget1.range, outgoingConversionFunction)


def range_converter(valueToConvert, startingRange, destinationRange):
    x1 = startingRange[0]
    x2 = startingRange[1]
    y1 = destinationRange[0]
    y2 = destinationRange[1]
    convertedValue = (y2-y1)/(x2-x1)*(valueToConvert-x1)+y1
    if type(y1) is int:
        convertedValue = round(convertedValue)
    return convertedValue


class _OutgoingConverter:
    def __init__(self, startingRange, destinationRange, function):
        self._startingRange = startingRange
        self._destinationRange = destinationRange
        self._function = function

    def convert(self, valueToConvert):
        return self._function(valueToConvert, self._startingRange, self._destinationRange)


class _LinkableWidget(ABC):
    def __init__(self, *, valueRange, **kwargs):
        super().__init__(**kwargs)
        self._isDriving = True
        self.range = valueRange
        self.outgoingConverter = _OutgoingConverter(None, None, lambda *_, **__: None)
        self.linkCommand = lambda *_, **__: None
        self._externalCommand = lambda *_, **__: None
        self._bind_callback(self._callback)

    def set_value(self, value, isDriving):
        self._isDriving = isDriving
        self._custom_set(value)
        self._isDriving = True

    def bind_external_command(self, command):
        self._externalCommand = command

    def _callback(self, *args):
        linkValue, externalCommandValue = self._custom_callback(*args)
        if self._isDriving and linkValue is not None:
            linkValue = self.outgoingConverter.convert(linkValue)
            self.linkCommand(linkValue, isDriving=False)
        if externalCommandValue is not None:
            self._externalCommand(externalCommandValue)

    @abstractmethod
    def _custom_callback(self, *args):
        # return linkValue and externalCommandValue
        raise NotImplementedError

    @abstractmethod
    def _custom_set(self, value):
        raise NotImplementedError

    @abstractmethod
    def _bind_callback(self, callback):
        raise NotImplementedError


class LinkableEntry(_LinkableWidget, tk.Entry):
    def __init__(self, master, valueRange=(-100, 100), initialValue=0, **kwargs):
        self._previousValue = initialValue
        self._previousValidNumber = initialValue
        self._value = tk.StringVar(value=self._previousValue)
        super().__init__(valueRange=valueRange, master=master, textvariable=self._value, **kwargs)
        self.bind('<FocusIn>', self._on_focus_in)
        self.bind('<FocusOut>', self._on_focus_out)
        self.bind('<<VirtualMouseWheel>>', self._on_mouse_wheel)

    @property
    def _valueIsValidNonNumeric(self):
        return self._value.get() in ['-', '']

    def _custom_callback(self, *_):
        linkValue = None
        externalCommandValue = None
        try:
            newValue = self._value.get()
            newValue = int(newValue)
            if self.range[0] <= newValue <= self.range[1]:
                self._value.set(newValue)
                self._previousValidNumber = newValue
                linkValue = newValue
            else:
                self._value.set(self._previousValue)
        except ValueError:
            if not self._valueIsValidNonNumeric:
                self._value.set(self._previousValue)
        self._previousValue = self._value.get()
        return linkValue, externalCommandValue

    def _custom_set(self, value):
        self._value.set(value)
        if self.focus_get() is self:
            self.select_range(0, tk.END)
            self.icursor(tk.END)

    def _bind_callback(self, callback):
        self._value.trace_add('write', callback)

    def _on_focus_in(self, event):
        self.select_range(0, tk.END)

    def _on_focus_out(self, event):
        if self._valueIsValidNonNumeric:
            self.set_value(self._previousValidNumber, isDriving=False)

    def _on_mouse_wheel(self, event):
        if self.focus_get() is self:
            if event.delta == 0:
                delta = event.state
            else:
                delta = event.delta
            if delta > 0:
                delta = 1
            else:
                delta = -1
            if not self._valueIsValidNonNumeric:
                value = self._value.get()
                self.set_value(int(value)+delta, isDriving=True)
            return 'break'  # prevent callback from being called twice


class LinkableSlider(_LinkableWidget, stl.ScaleAutoStyle):
    def __init__(self, master, **kwargs):
        super().__init__(valueRange=(0., 1.,), master=master, **kwargs)
        self.bind('<Button-1>', self._set_slider_to_click_position)

    def _custom_callback(self, *args):
        event = args[0]
        linkValue = float(event)
        externalCommandValue = event
        return linkValue, externalCommandValue

    def _custom_set(self, value):
        self.set(value)

    def _bind_callback(self, callback):
        self.config(command=callback)

    def _set_slider_to_click_position(self, event):
        self.event_generate('<Button-3>', x=event.x, y=event.y)
        return 'break'  # ensures that original Button-1 behavior is not called
