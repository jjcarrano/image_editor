import cv2
import numpy as np
import enum
from numba import njit, prange
import copy
from scipy import fft


class ParamType(enum.Enum):
    EQUALIZE = enum.auto()
    BRIGHTNESS = enum.auto()
    CONTRAST = enum.auto()
    SATURATION = enum.auto()
    WARMTH = enum.auto()
    TINT = enum.auto()
    TWO_TONE_HUE = enum.auto()
    TWO_TONE_SATURATION = enum.auto()


class _Observable:
    def __init__(self, initialValue=None):
        self._data = initialValue
        self._callbacks = set()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._do_callbacks()

    def add_callback(self, func):
        self._callbacks.add(func)

    def _do_callbacks(self):
        for func in self._callbacks:
            func(self._data)


class _UniquePixelData:
    # consider emulating matlab structure arrays
    def __init__(self, pixels):
        uniquePixels, ind, counts = self._find_unique_rows(pixels.reshape((-1, 3)))
        self.values = uniquePixels
        self._reverseMapping = ind
        self.counts = counts
        self._inputShape = pixels.shape

    def reverse(self, asUint8=True):
        values = self.values
        if values.dtype != np.uint8 and asUint8:
            values = (values*255+.5).astype(np.uint8)
        return self._jit_reverse(values, self._reverseMapping, self._inputShape)

    @staticmethod
    @njit(cache=True)
    def _jit_reverse(values, reverseMapping, shape):
        # JIT'd methods must be made static
        return values[reverseMapping].reshape(shape)

    def _find_unique_rows(self, inputArr):
        # This function is necessary because numpy.unique is not fully compatible with tkinter and multithreading.
        # It's also several times faster.
        sortInd = np.lexsort(inputArr.T)
        return self._jit_find_unique_rows(inputArr, sortInd)

    @staticmethod
    @njit(cache=True)
    def _jit_find_unique_rows(inputArr, sortInd):
        uniqueRows = inputArr.copy()
        uniqueRows[0] = inputArr[sortInd[0]]
        uniqueRowIndex = 0
        counts = np.ones(inputArr.shape[0], dtype=np.int64)
        reverseMap = np.zeros(inputArr.shape[0], dtype=np.int64)
        for i in range(1, inputArr.shape[0]):
            row = sortInd[i]
            for j in range(inputArr.shape[1]):
                if inputArr[row, j] != uniqueRows[uniqueRowIndex, j]:
                    uniqueRowIndex += 1
                    uniqueRows[uniqueRowIndex] = inputArr[row]
                    break
            else:
                counts[uniqueRowIndex] += 1
            reverseMap[row] = uniqueRowIndex
        uniqueRowIndex += 1
        return uniqueRows[:uniqueRowIndex], reverseMap, counts[:uniqueRowIndex]


class _RgbModifier:
    def __init__(self, modifiedPixels):
        self._modifiedPixels = modifiedPixels
        histogram = self._generate_image_histogram(self._modifiedPixels.values, self._modifiedPixels.counts)
        self.cdf = np.cumsum(histogram)
        self.histogramFrequencies = fft.dct(histogram)
        self.x = np.arange(256)
        self.xSqr = (np.pi/256*self.x)**2

    def equalize(self, t):
        diffusedHistogram = fft.idct(self.histogramFrequencies*np.exp(-self.xSqr*t**2))
        if t >= 0:
            new_pixels = np.interp(self.cdf, np.cumsum(diffusedHistogram), self.x).astype(np.float32)
        else:
            new_pixels = np.interp(np.cumsum(diffusedHistogram), self.cdf, self.x).astype(np.float32)
        self._modifiedPixels.values = np.reshape(new_pixels[self._modifiedPixels.values.ravel()],
                                                 self._modifiedPixels.values.shape)

    @staticmethod
    @njit(parallel=True, cache=True)
    def _generate_image_histogram(values, counts):
        y = np.zeros(256, dtype=np.int64)
        for m in range(values.shape[0]):
            for n in range(values.shape[1]):
                y[values[m, n]] += counts[m]
        return y


class _LmsModifier:
    # Brightness, contrast, and white balance adjustments done in LMS space
    def __init__(self, modifiedPixels):
        self._modifiedPixels = modifiedPixels
        self.Ma = np.array([[0.95083895, -0.72104359, 0.03569572],
                            [0.28298905, 1.64690509, -0.0628575],
                            [-0.17145057, 0.03527366, 0.94478947]], dtype=np.float32)
        self.lsrgb2xyzMatrix = np.array([[0.4123908, 0.21263901, 0.01933082],
                                         [0.35758434, 0.71516868, 0.11919478],
                                         [0.18048079, 0.07219232, 0.95053215]])
        self.lsrgb2lmsMatrix = (self.lsrgb2xyzMatrix@self.Ma).astype(np.float32)
        self.lms2lsrgbMatrix = np.linalg.inv(self.lsrgb2xyzMatrix@self.Ma).astype(np.float32)

    @staticmethod
    @njit(parallel=True, cache=True)
    def _srgb2lms(pixels, lsrgb2lmsMatrix):
        for m in prange(pixels.shape[0]):
            for n in prange(pixels.shape[1]):
                subPixel = pixels[m, n]
                if subPixel <= 10.31475:
                    pixels[m, n] = subPixel/3294.6
                else:
                    pixels[m, n] = ((subPixel+14.025)/269.025)**2.4
        pixels = pixels@lsrgb2lmsMatrix
        return pixels

    def modify_brightness_contrast_wb(self, brightness, contrast, warmth, tintFactor):
        brightness **= 2.2
        self._modifiedPixels.values = self._srgb2lms(self._modifiedPixels.values, self.lsrgb2lmsMatrix)
        inflectionPoint = self._determine_inflection_point(
            self._modifiedPixels.values, self._modifiedPixels.counts, brightness)
        whiteBalanceScale = self._determine_white_balance_scale(warmth, tintFactor)
        self._bezier_transform(self._modifiedPixels.values,
                               inflectionPoint, brightness*whiteBalanceScale, contrast)

    @staticmethod
    @njit(parallel=True, cache=True)
    def _determine_inflection_point(values, counts, brightness):
        brightness = brightness**(1/3)
        valueAccumulator = 0
        countAccumulator = 0
        for m in prange(values.shape[0]):
            for n in prange(values.shape[1]):
                brightnessAdjustedValue = brightness*values[m, n]**(1/3)
                if brightnessAdjustedValue > 1:
                    brightnessAdjustedValue = 1
                valueAccumulator += brightnessAdjustedValue*counts[m]
                countAccumulator += counts[m]
        meanEstimate = valueAccumulator/countAccumulator
        return max(min(meanEstimate, .9), .1)**3

    def _determine_white_balance_scale(self, warmth, tintFactor):
        mired = -246.19488086865348*warmth+153.80511913134652
        tint = 0.05682694697114664*tintFactor+0.003173053028853357
        maxTint = (-1.09163980891296e-14*mired**5+
                   2.67891153048463e-11*mired**4+
                   -2.66040681064267e-8*mired**3+
                   1.35844899211827e-5*mired**2+
                   -3.67404502310367e-3*mired+
                   .437477278353160)
        tint = min(tint, maxTint)
        sourceWhitePoint = self._calculate_source_white_point(mired, tint)
        return 1/(sourceWhitePoint@self.Ma)

    @staticmethod
    def _calculate_source_white_point(mired, tint):
        if mired > 40:
            a = (-.2661239, -.2661239, -3.0258469)
            b = (-.2343589, -.2343589, 2.1070379)
            c = (.8776956, .8776956, .2226347)
            d = (.179910, .179910, .240390)
            e = (-1.1063814, -.9549476, 3.0817580)
            f = (-1.3481102, -1.37418593, -5.87338670)
            g = (2.18555832, 2.09137015, 3.75112997)
            h = (-.20219683, -.16748867, -.37001483)
            if mired > 1000000/2222:
                i = 0
            elif mired > 1000000/4000:
                i = 1
            else:
                i = 2
            x = a[i]/10**9*mired**3+b[i]/10**6*mired**2+c[i]/10**3*mired+d[i]
            y = e[i]*x**3+f[i]*x**2+g[i]*x+h[i]
            interU = -12*(8*e[i]*x**3+4*f[i]*x**2+-4*h[i]-1)
            interV = -6*(4*e[i]*x**3+(2*f[i]-9*e[i])*x**2+-6*f[i]*x+-3*g[i]-2*h[i])
            normVectMag = tint/np.sqrt(interU**2+interV**2)
            u = 4*x/(12*y-2*x+3)-interV*normVectMag
            v = 6*y/(12*y-2*x+3)+interU*normVectMag
        else:
            u = 9.20087111940772e-5*mired+0.179201141114284-tint*0.9536545649322323
            v = 0.000291603457752117*mired+0.262421009337607+tint*0.3009035905134979
        X = 3*u/(2*v)
        Z = (-u-10*v+4)/(2*v)
        return np.array([X, 1., Z])

    @staticmethod
    @njit(parallel=True, cache=True)
    def _bezier_transform(pixels, inflectionPoint, brightness, contrast):
        invBc = 1/(brightness*contrast)
        for m in prange(pixels.shape[0]):
            for n in prange(pixels.shape[1]):
                subPixel = pixels[m, n]
                if subPixel <= inflectionPoint/brightness[n]:
                    x0 = inflectionPoint*(1-contrast)
                    x1 = max(inflectionPoint*(2-contrast)/2, 0)
                    x2 = (3*x1+inflectionPoint)/4
                    x3 = inflectionPoint
                    y0 = 0
                    y3 = x3
                else:
                    x0 = inflectionPoint
                    x2 = min(contrast*(brightness[n]-inflectionPoint)/2+inflectionPoint, 1)
                    x1 = (3*x2+inflectionPoint)/4
                    x3 = brightness[n]*contrast+inflectionPoint*(1-contrast)
                    y0 = inflectionPoint
                    y3 = 1
                a = (-x0+3*x1-3*x2+x3)*invBc[n]
                b = (3*x0-6*x1+3*x2)*invBc[n]
                c = (-3*x0+3*x1)*invBc[n]
                d = inflectionPoint/brightness[n]+(x0-inflectionPoint)*invBc[n]-subPixel
                delta0 = b**2-3*a*c
                delta1 = 2*b**3-9*a*b*c+27*a**2*d
                bigC = np.cbrt((delta1+np.sqrt(delta1**2-4*delta0**3))*.5)
                t = -1/(3*a)*(b+bigC+delta0/bigC)
                pixels[m, n] = (1-t)**3*y0+3*(1-t)**2*t*x1+3*(1-t)*t**2*x2+t**3*y3


class _OklabModifier:
    # Saturation adjustment done in OKLAB space
    def __init__(self, modifiedPixels):
        self._modifiedPixels = modifiedPixels
        self.lms2oklmsMatrix = np.array([[0.90929928, 0.4085416, 0.14721228],
                                         [0.06450354, 0.49764315, 0.16153364],
                                         [0.02619717, 0.09381525, 0.69125408]], dtype=np.float32)
        self.lmsPrime2oklabMatrix = np.array([[0.2104542553, 1.9779984951, 0.0259040371],
                                              [0.7936177850, -2.4285922050, 0.7827717662],
                                              [-0.0040720468, 0.4505937099, -0.8086757660]])
        self.oklab2lmsPrimeMatrix = np.array([[0.9999999985, 1.0000000089, 1.0000000547],
                                              [0.3963377922, -0.1055613423, -0.0894841821],
                                              [0.2158037581, -0.0638541748, -1.2914855379]])
        self.oklms2lsrgbMatrix = np.array([[4.07674166, -1.268438, -0.00419609],
                                           [-3.30771159, 2.6097574, -0.70341861],
                                           [0.23096993, -0.3413194, 1.7076147]], dtype=np.float32)

    def modify_hue_saturation(self, saturationFactor, twoToneHue, twoToneSaturation):
        lmsPrime = self._lms2lmsPrime(self._modifiedPixels.values, self.lms2oklmsMatrix)
        theta = twoToneHue*np.pi/180+np.pi/2
        a = twoToneSaturation-1
        b = saturationFactor*a*np.sin(2*theta)/2
        adjustmentMatrix = np.array([[1., 0., 0.],
                                     [0., saturationFactor*(1+a*np.cos(theta)**2), b],
                                     [0., b, saturationFactor*(1+a*np.sin(theta)**2)]])
        adjustedLmsPrime = lmsPrime@(self.lmsPrime2oklabMatrix @
                                     adjustmentMatrix @
                                     self.oklab2lmsPrimeMatrix).astype(np.float32)
        self._modifiedPixels.values = self._lmsPrime2srgb(adjustedLmsPrime, self.oklms2lsrgbMatrix)

    @staticmethod
    @njit(parallel=True, cache=True)
    def _lms2lmsPrime(lms, lms2oklmsMatrix):
        lmsPrime = lms@lms2oklmsMatrix
        for m in prange(lmsPrime.shape[0]):
            for n in prange(lmsPrime.shape[1]):
                if lmsPrime[m, n] < 0:
                    lmsPrime[m, n] = 0
                else:
                    lmsPrime[m, n] **= 1/3
        return lmsPrime

    @staticmethod
    @njit(parallel=True, cache=True)
    def _lmsPrime2srgb(oklms, oklms2lsrgbMatrix):
        for m in prange(oklms.shape[0]):
            for n in prange(oklms.shape[1]):
                oklms[m, n] **= 3
        srgb = oklms@oklms2lsrgbMatrix
        for m in prange(srgb.shape[0]):
            for n in prange(srgb.shape[1]):
                subPixel = srgb[m, n]
                if subPixel < 0:
                    srgb[m, n] = 0
                elif subPixel > 1:
                    srgb[m, n] = 1
                elif subPixel <= .0031308:
                    srgb[m, n] = subPixel*12.92
                else:
                    srgb[m, n] = 1.055*subPixel**(1/2.4)-.055
        return srgb


class _ImageProcessor:
    def __init__(self, image):
        self._originalPixels = _UniquePixelData(image)
        self._modifiedPixels = copy.deepcopy(self._originalPixels)
        self._rgbModifier = _RgbModifier(self._modifiedPixels)
        self._lmsModifier = _LmsModifier(self._modifiedPixels)
        self._oklchModifier = _OklabModifier(self._modifiedPixels)
        self._processedImage = _Observable(self._modifiedPixels.reverse())
        self.processingParams = {ParamType.EQUALIZE: 0.,
                                 ParamType.BRIGHTNESS: 1.,
                                 ParamType.CONTRAST: 1.,
                                 ParamType.SATURATION: 1.,
                                 ParamType.WARMTH: 0.,
                                 ParamType.TINT: 0.,
                                 ParamType.TWO_TONE_HUE: 0.,
                                 ParamType.TWO_TONE_SATURATION: 1.}

    @property
    def processedImage(self):
        return self._processedImage.data

    def change_processing_params(self, paramDict):
        for param, value in paramDict.items():
            self.processingParams[param] = value
        self._process_image()

    def add_processedImage_callback(self, func):
        self._processedImage.add_callback(func)

    def _process_image(self):
        self._modifiedPixels.values = self._originalPixels.values.copy()
        self._rgbModifier.equalize(self.processingParams[ParamType.EQUALIZE])
        self._lmsModifier.modify_brightness_contrast_wb(self.processingParams[ParamType.BRIGHTNESS],
                                                        self.processingParams[ParamType.CONTRAST],
                                                        self.processingParams[ParamType.WARMTH],
                                                        self.processingParams[ParamType.TINT])
        self._oklchModifier.modify_hue_saturation(self.processingParams[ParamType.SATURATION],
                                                  self.processingParams[ParamType.TWO_TONE_HUE],
                                                  self.processingParams[ParamType.TWO_TONE_SATURATION])
        self._processedImage.data = self._modifiedPixels.reverse()


class Model:
    def __init__(self, filePath, maxDisplayImageSize=(780, 1525)):

        def downscale_image_if_too_big(img):
            maxRelDim = np.maximum(img.shape[0]/maxDisplayImageSize[0], img.shape[1]/maxDisplayImageSize[1])
            if maxRelDim > 1:
                # add anti-aliasing filter before downsampling, consider linearizing first
                img = cv2.resize(img, None, fx=maxRelDim**-1, fy=maxRelDim**-1, interpolation=cv2.INTER_CUBIC)
            return img

        self.filePath = filePath
        self._existUnsavedChanges = _Observable(False)
        image = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
        # Add compatibility for 16 and 32 bit images
        if image.dtype == np.uint16:
            image = np.round(image/257).astype(np.uint8)
        elif image.dtype == np.float32:
            image = np.round(image*255).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise TypeError('Can\'t handle image of type '+str(image.dtype))
        image = image[:, :, [2, 1, 0]]
        self._originalTrueImage = image
        image = downscale_image_if_too_big(image)
        self.originalDisplayImage = image
        self._displayImageProcessor = _ImageProcessor(image)

    @property
    def processedDisplayImage(self):
        return self._displayImageProcessor.processedImage

    @property
    def existUnsavedChanges(self):
        return self._existUnsavedChanges.data

    def change_processing_params(self, paramDict):
        self._existUnsavedChanges.data = True
        self._displayImageProcessor.change_processing_params(paramDict)

    def save_image(self, filePath):
        processingParams = copy.deepcopy(self._displayImageProcessor.processingParams)
        self._existUnsavedChanges.data = False
        trueImageProcessor = _ImageProcessor(self._originalTrueImage)
        trueImageProcessor.change_processing_params(processingParams)
        cv2.imwrite(filePath, trueImageProcessor.processedImage[:, :, [2, 1, 0]])

    def add_processedImage_callback(self, func):
        self._displayImageProcessor.add_processedImage_callback(func)

    def add_existUnsavedChanges_callback(self, func):
        self._existUnsavedChanges.add_callback(func)


Model('tile.png').change_processing_params({ParamType.BRIGHTNESS: 1.1})
