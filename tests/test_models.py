from src.models import _UniquePixelData
import numpy as np


class TestUniquePixelData:
    def test__init__(self):
        pixels = np.random.randint(256, size=(10000, 3), dtype=np.uint8)
        uniquePixelData = _UniquePixelData(pixels)
        uniquePixelsNumpy = np.unique(pixels, axis=0)
        # conditions required for uniquePixelData.values to be equivalent to np.unique output ignoring row order
        passConditions = [
            np.array_equal(np.unique(uniquePixelData.values, axis=0), uniquePixelsNumpy),
            uniquePixelData.values.shape == uniquePixelsNumpy.shape
        ]
        assert all(passConditions)

    def test_reverse(self):
        pixels = np.random.randint(256, size=(10000, 3), dtype=np.uint8)
        uniquePixelData = _UniquePixelData(pixels)
        assert np.array_equal(pixels, uniquePixelData.reverse())
