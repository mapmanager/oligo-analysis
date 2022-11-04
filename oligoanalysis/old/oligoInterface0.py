"""
"""
import numpy as np

import napari

from oligoAnalysis import oligoAnalysisFolder
from oligoUtils import getOtsuThreshold

def viewRgbMask(folderPath):
    oa = oligoAnalysisFolder(folderPath)
    print(oa._dfFolder)

    rgbStack = oa.loadRgbStack()
    
    mergedMask = oa._getCellPoseMask()
    print('  mergedMask:', mergedMask.shape, mergedMask.dtype)

    viewer = napari.Viewer()

    rgbStackLayer = viewer.add_image(rgbStack, name='rgb merged')
    rgbMaskLayer = viewer.add_labels(mergedMask, name='rgb merged mask')

    napari.run()

def checkGaussian():
    """Not sure if sigma=1 or sigma=(0,1,1) is better?
    """
    folderPath = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/FST'
    
    #viewRgbMask(folderPath)

    oaf = oligoAnalysisFolder(folderPath)

    firstFile = oaf._dfFolder['file'].tolist()[0]
    imgRed = oaf.getColorChannel(firstFile, 'red')  # uint16

    # I prefer sigma=1 rather than sigma=(0,1,1)
    otsuThreshold, gaussianSigma, img_blurred, img_binary = getOtsuThreshold(imgRed, sigma = 1)
    # otsuThreshold2, img_blurred2, img_binary2 = getOtsuThreshold(imgRed, sigma = 1)

    print('  imgRed:', imgRed.shape, np.min(imgRed), np.max(imgRed), imgRed.dtype)
    
    print('  otsuThreshold:', otsuThreshold)
    print('  gaussianSigma:', gaussianSigma)
    print('  img_blurred:', img_blurred.shape, np.min(img_blurred), np.max(img_blurred), img_blurred.dtype)
    print('  img_binary:', img_binary.shape, np.min(img_binary), np.max(img_binary), img_binary.dtype)
    
    # print('  otsuThreshold:', otsuThreshold2)
    # print('  img_blurred:', img_blurred2.shape, np.min(img_blurred2), np.max(img_blurred2), img_blurred2.dtype)
    # print('  img_binary:', img_binary2.shape, np.min(img_binary2), np.max(img_binary2), img_binary2.dtype)
    
    viewer = napari.Viewer()
    
    imgLayer = viewer.add_image(imgRed, name='imgRed')
    imgLayer.colormap = 'red'
    #imgLayer.contrast_limits = (0, 128)

    blurredLayer = viewer.add_image(img_blurred, name='img_blurred')
    binaryLayer = viewer.add_labels(img_binary, name='img_binary')

    # blurredLayer2 = viewer.add_image(img_blurred2, name='img_blurred2')
    # binaryLayer2 = viewer.add_labels(img_binary2, name='img_binary2')

    napari.run()

if __name__ == '__main__':
    folderPath = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/FST'
    
    #viewRgbMask(folderPath)

    oaf = oligoAnalysisFolder(folderPath)

    fileIdx = 1
    oneFileName = oaf._dfFolder['file'].tolist()[fileIdx]

    #imgRed = oa.getColorChannel(firstFile, 'red', astype=np.uint8)  # uint16

    oaf.loadRgbStack()
    
    # get the cellpose mask for one file
    mergedMask = oaf._getCellPoseMask()
    imgMask, imgRed, imgGreen = oaf._getCellPoseMask_file(oneFileName)

    print('  imgRed:', imgRed.shape, np.min(imgRed), np.max(imgRed), imgRed.dtype)
    print('  imgGreen:', imgGreen.shape, np.min(imgGreen), np.max(imgGreen), imgGreen.dtype)

    # I prefer sigma=1 rather than sigma=(0,1,1)
    otsuThreshold, gaussianSigma, imgRed_blurred, imgRed_binary = \
                            getOtsuThreshold(imgRed, sigma = 1)
    
    print('  otsuThreshold:', otsuThreshold)
    print('  img_blurred:', imgRed_blurred.shape, np.min(imgRed_blurred), np.max(imgRed_blurred), imgRed_blurred.dtype)
    print('  img_binary:', imgRed_binary.shape, np.min(imgRed_binary), np.max(imgRed_binary), imgRed_binary.dtype)
        
    viewer = napari.Viewer()
    
    imgRedLayer = viewer.add_image(imgRed, name='imgRed')
    imgRedLayer.colormap = 'red'
    imgRedLayer.contrast_limits = (0, 150)

    imgGreenLayer = viewer.add_image(imgGreen, name='imgGreen')
    imgGreenLayer.colormap = 'green'
    imgGreenLayer.contrast_limits = (0, 150)

    #blurredLayer = viewer.add_image(img_blurred, name='img_blurred')
    binaryLayer = viewer.add_labels(imgRed_binary, name='imgRed_binary')

    imgMask_layer = viewer.add_labels(imgMask, name='imgMask')

    napari.run()
