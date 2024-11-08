import numpy as np


def srgb2linear_srgb(srgb):
    srgb = srgb.copy()
    boolArr = srgb <= .04045
    srgb[boolArr] = srgb[boolArr]/12.92
    srgb[~boolArr] = ((srgb[~boolArr]+.055)/1.055)**2.4
    return srgb


def srgb2xyz(srgb):
    srgb = srgb2linear_srgb(srgb)
    transMatrix = np.array([[0.4123908, 0.21263901, 0.01933082],
                            [0.35758434, 0.71516868, 0.11919478],
                            [0.18048079, 0.07219232, 0.95053215]])
    return srgb@transMatrix


def xyz2oklab(xyz):
    transMatrix1 = np.array([[0.81902244, 0.03298366, 0.04817719],
                             [0.36190626, 0.92928685, 0.26423953],
                             [-0.12887379, 0.03614467, 0.63354783]])
    transMatrix2 = np.array([[0.2104542553, 1.9779984951, 0.0259040371],
                             [0.7936177850, -2.4285922050, 0.7827717662],
                             [-0.0040720468, 0.4505937099, -0.8086757660]])
    return ((xyz@transMatrix1)**(1/3))@transMatrix2


def srgb2oklab(srgb):
    srgb = srgb2linear_srgb(srgb)
    transMatrix1 = np.array([[0.4122214708, 0.2119034982, 0.0883024619],
                             [0.5363325363, 0.6806995451, 0.2817188376],
                             [0.0514459929, 0.1073969566, 0.6299787005]])
    transMatrix2 = np.array([[0.2104542553, 1.9779984951, 0.0259040371],
                             [0.7936177850, -2.4285922050, 0.7827717662],
                             [-0.0040720468, 0.4505937099, -0.8086757660]])
    return ((srgb@transMatrix1)**(1/3))@transMatrix2


def linear_srgb2srgb(srgb):
    srgb = srgb.copy()
    boolArr = srgb <= .0031308
    srgb[boolArr] = srgb[boolArr]*12.92
    srgb[~boolArr] = 1.055*srgb[~boolArr]**(1/2.4)-.055
    return srgb


def xyz2srgb(xyz):
    transMatrix = np.array([[3.24096994, -0.96924364, 0.05563008],
                            [-1.53738318, 1.8759675, -0.20397696],
                            [-0.49861076, 0.04155506, 1.05697151]])
    srgb = xyz@transMatrix
    return linear_srgb2srgb(srgb)


def oklab2xyz(oklab):
    transMatrix1 = np.array([[0.9999999985, 1.0000000089, 1.0000000547],
                             [0.3963377922, -0.1055613423, -0.0894841821],
                             [0.2158037581, -0.0638541748, -1.2914855379]])
    transMatrix2 = np.array([[1.22687988, -0.04057575, -0.07637293],
                             [-0.55781501, 1.11228682, -0.42149334],
                             [0.28139107, -0.07171107, 1.58692402]])
    return ((oklab@transMatrix1)**3)@transMatrix2


def oklab2srgb(oklab):
    transMatrix1 = np.array([[0.9999999985, 1.0000000089, 1.0000000547],
                             [0.3963377922, -0.1055613423, -0.0894841821],
                             [0.2158037581, -0.0638541748, -1.2914855379]])
    transMatrix2 = np.array([[4.07674166, -1.268438, -0.00419609],
                             [-3.30771159, 2.6097574, -0.70341861],
                             [0.23096993, -0.3413194, 1.7076147]])
    srgb = ((oklab@transMatrix1)**3)@transMatrix2
    return linear_srgb2srgb(srgb)
