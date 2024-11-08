from src.models import _UniquePixelData
import numpy as np


def test_UniquePixelData():
    pixels = np.random.randint(256, size=(1000, 3), dtype=np.uint8)
    uniquePixelData = _UniquePixelData(pixels)
    uniquePixelsNumpy = np.unique(pixels, axis=0)
    # conditions required for uniquePixelData.values to be equivalent to np.unique output ignoring row order
    passConditions = [
        np.array_equal(np.unique(uniquePixelData.values, axis=0), uniquePixelsNumpy),
        uniquePixelData.values.shape == uniquePixelsNumpy.shape
    ]
    assert all(passConditions)
