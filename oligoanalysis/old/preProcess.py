"""
Load folders of manually saved .tif  and export a csv with file database.

Including
    - filename
    - pixel x/y/z
    - voxel x/y/z
"""

import os, sys
#import glob
#import pathlib
from pprint import pprint

import numpy as np
import pandas as pd
import scipy

import tifffile

import matplotlib.pyplot as plt
import seaborn as sns

def loadMask():
    path = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/cellpose-model/fst-concat-small-slices/fst-concat-rgb-small_seg.npy'
    maskData = np.load(path, allow_pickle=True)
    maskItem = maskData.item()
    print(maskItem.shape, maskItem.dtype)

def getSeg_npy(tifPath):
    """Given an original tif path, get the cellpose '_seg.npy' file.
    """
    _path, _file = os.path.split(tifPath)
    segFile = os.path.splitext(_file)[0] + '_seg.npy'
    segPath = os.path.join(_path, segFile)
    return segPath

def loadSegMask_npy(tifPath):
    """Given a tif file, load masks from cellpose seg npy file.
    """
    segPath = getSeg_npy(tifPath)

    dat = np.load(segPath, allow_pickle=True).item()
    masks = dat['masks']
    return masks

def plotCellpose():
    """Load a cellpose segmentation and save the mask. Mask is a labeled stack.
    """
    #  1) in fiji convert to tif
    #  2) run makeRgbStack() on folder to make a concatenated rgb stack
    #  3) open stack in cellpose and run the model on it
    #  4) save the resulting npy file

    # original merged tif (small)
    
    # what the model was trained on
    tifPath = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/cellpose-model/fst-concat-small-slices/fst-concat-rgb-small.tif'

    # after training, ran saved model on this
    tifPath = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/Morphine/Morphine-rgb-merged-small.tif'

    imageData = tifffile.imread(tifPath)
    print('imageData:', imageData.shape, imageData.dtype)  # (46, 196, 196, 3)

    # what we saved in cellpose after running model on original tif
    # get the '_seg.npy' file from original tif
    #segPath = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/cellpose-model/fst-concat-small-slices/fst-concat-rgb-small_seg.npy'
    _path, _file = os.path.split(tifPath)
    segFile = os.path.splitext(_file)[0] + '_seg.npy'
    segPath = os.path.join(_path, segFile)

    # create a save path for mask (from npy file)
    _folder, _filename = os.path.split(segPath)
    maskFileName = os.path.splitext(_filename)[0] + '_mask.tif'
    saveMaskPath = os.path.join(_folder, maskFileName)

    from cellpose import plot, utils
    
    dat = np.load(segPath, allow_pickle=True).item()

    # dat.keys() are
    # ['outlines', 'colors', 'masks', 'current_channel', 'filename',
    #         'flows', 'zdraw', 'model_path', 'flow_threshold', 'cellprob_threshold']
    
    # masks are labeled data with 0 as background and (1, 2, 3, ...) for each mask
    print('dat.keys():', dat.keys())
    _masks = dat['masks']
    print('_masks:', type(_masks), _masks.shape, _masks.dtype, 'min:', np.min(_masks), 'max:', np.max(_masks))

    print('saving saveMaskPath:', saveMaskPath)
    tifffile.imwrite(saveMaskPath, _masks, imagej=True)

    '''
    # plot image with masks overlaid
    # mask_RGB = plot.mask_overlay(dat['img'], dat['masks'],
    #                         colors=np.array(dat['colors']))
    mask_RGB = plot.mask_overlay(imageData, dat['masks'],
                            colors=np.array(dat['colors']))
    print('mask_RGB:', type(mask_RGB), mask_RGB.shape, mask_RGB.dtype)
    for i in range(3):
        plt.plot(mask_RGB[:,:,i])
    plt.show()

    # plot image with outlines overlaid in red
    if 0:
        outlines = utils.outlines_list(dat['masks'])
        plt.imshow(dat['img'])
        for o in outlines:
            plt.plot(o[:,0], o[:,1], color='r')
    '''

def postAnalysis():
    """
    Load cellpose mask/labels.
    For each mask/label
        1) dilate 1-2 pixels
        2) sum the red channel within the mask
        3) decide if it is an oligo or not!
    """
    # what the model was trained on
    tifPath = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/cellpose-model/fst-concat-small-slices/fst-concat-rgb-small.tif'
    
    # model was applided to this
    tifPath = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/Morphine/Morphine-rgb-merged-small.tif'
    
    imageData = tifffile.imread(tifPath)
    print('imageData:', imageData.shape, imageData.dtype)

    masks = loadSegMask_npy(tifPath)
    print('  masks:', masks.shape, masks.dtype)

    numMasks = np.max(masks)
    print('  numMasks:', numMasks)
    greenMaskSumList = [0]
    redMaskSumList = [0]
    numPlots = 0  # for debug
    for maskIdx in range(numMasks):
        if maskIdx == 0:
            # background
            continue
        _oneMask = masks == maskIdx  # (46, 196, 196)
        #print('_oneMask:', type(_oneMask), _oneMask.shape, _oneMask.dtype)

        # dilate the mask
        dilatedMask = scipy.ndimage.binary_dilation(_oneMask, iterations=3)
        #print('dilatedMask:', type(dilatedMask), dilatedMask.shape, dilatedMask.dtype)

        # make a ring ?, will change percentRedThrehold
        dilatedMask = dilatedMask ^ _oneMask

        # oligo channel
        redChannelIdx = 0
        imageData_red = imageData[:,:,:,redChannelIdx]
        # dapi channel
        greenChannelIdx = 1
        imageData_green = imageData[:,:,:,greenChannelIdx]

        # imagedata is (46, 196, 196, 3)
        redImageMask = np.where(dilatedMask>0, imageData_red, 0)  # 0 is fill value
        greenImageMask = np.where(dilatedMask>0, imageData_green, 0)  # 0 is fill value

        # get the number of pixels in the mask
        dilatedMaskCount = np.count_nonzero(dilatedMask)

        redImageMaskSum = np.sum(redImageMask) / dilatedMaskCount
        redMaskSumList.append(redImageMaskSum)

        greenImageMaskSum = np.sum(greenImageMask) / dilatedMaskCount
        greenMaskSumList.append(greenImageMaskSum)

        doPlot = True
        percentRedThrehold = 6  # for original trained data
        percentRedThrehold = 9  # for 'morphine' trained on the model
        if doPlot and redImageMaskSum > percentRedThrehold:

            # check we have the correct channel !
            # if True:
            #     # _tmpMax = np.max(imageData_red, axis=0)
            #     # plt.imshow(_tmpMax, cmap=plt.cm.Reds, alpha=0.7)
            #     _tmpMax = np.max(imageData_green, axis=0)
            #     plt.imshow(_tmpMax, cmap=plt.cm.Greens, alpha=0.7)
            #     _tmpMax = np.max(dilatedMask, axis=0)
            #     plt.imshow(_tmpMax, cmap=plt.cm.Blues, alpha=0.3)
            #     plt.show()

            # make z-projections and plot
            if True or numPlots < 10:
                print(f'== maskIdx:{maskIdx}')
                redMaxProject = np.max(redImageMask, axis=0)
                greenMaxProject = np.max(greenImageMask, axis=0)
                
                # dilatedMaskMax = np.max(dilatedMask, axis=0)
                # dilatedMaskMax = dilatedMaskMax / np.max(dilatedMaskMax) * 30  # force it to be dim
                # dilatedMaskMax = dilatedMaskMax.astype(np.uint8)
                
                '''
                # rgb plot is way too dim
                rgbMax = np.zeros((dilatedMaskMax.shape[0], dilatedMaskMax.shape[1], 3), dtype=np.uint8)  # (m, n, 3)
                rgbMax[:,:,0] = redMax
                rgbMax[:,:,1] = greenMax
                rgbMax[:,:,2] = dilatedMaskMax
                rgbMax = rgbMax.astype(np.uint8)
                plt.imshow(rgbMax)
                '''

                redMaxProject_min = np.min(np.where(redMaxProject>0))
                redMaxProject_max = np.max(np.where(redMaxProject>0))
                print(f'  redMaxProject:{redMaxProject.shape} redMaxProject_min:{redMaxProject_min} redMaxProject_max:{redMaxProject_max}')
                plt.imshow(redMaxProject, cmap=plt.cm.Reds, vmin=redMaxProject_min, vmax=redMaxProject_max, alpha=0.7)

                greenMaxProject_min = np.min(np.where(greenMaxProject>0))
                greenMaxProject_max = np.max(np.where(greenMaxProject>0))
                print(f'  greenMaxProject_min:{greenMaxProject_min} greenMaxProject_max:{greenMaxProject_max}')
                #plt.imshow(greenMaxProject, cmap=plt.cm.Greens, vmin=greenMaxProject_min, vmax=greenMaxProject_max, alpha=0.7)

                plt.show()

            numPlots += 1
    #
    sns.histplot(x=redMaskSumList)
    #sns.histplot(x=greenMaskSumList)
    plt.show()

def makeMyMask():
    """Given one color channel, make a mask.
    
    Details:
        path here is rgb stack used in cellpose.
            1) load 3d 2-channel path and pull out one channel
            2) median filter (maybe)
            3) otsu filter
            4) ** view results
            5) save
    """
    tifPath = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/Morphine/Morphine-rgb-merged-small.tif'
    
    imageData = tifffile.imread(tifPath)
    print('imageData:', imageData.shape, imageData.dtype)
 
    # oligo channel
    redChannelIdx = 0
    imageData_red = imageData[:,:,:,redChannelIdx]  # (slice, x, y)
    print('imageData_red:', imageData_red.shape, imageData_red.dtype)


def makeRgbStack():
    """Given a folder of tif files, make one concatenated 3d stack.
    
    Each tif is 2-channel z-stack volume. The 3rd color channel (blue) in output will be 0.
    """
    
    path = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/Morphine'
    path = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/Morphine'
    
    files = os.listdir(path)
    fileIdx = 0
    totalNumSlices = 0
    rgbDataFinal = None
    for idx, file in enumerate(files):
        if not file.endswith('.tif'):
            continue
        if file.endswith('-rgb-merged.tif'):
            continue
        tifPath = os.path.join(path, file)
        tifData = tifffile.imread(tifPath)

        print(fileIdx, file)
        print('  tifData:', type(tifData), tifData.shape, tifData.dtype)
        _shape = tifData.shape  # (9, 2, 784, 784) (slices, channels, x, y)
        _slices = _shape[0]
        _channels = _shape[1]
        _x = _shape[2]
        _y = _shape[3]
        _dtype = tifData.dtype
        _rgbShape = (_slices, 3, _x, _y)
        rgbData = np.zeros(_rgbShape, _dtype)
        rgbData[:, 0:2, :, :] = tifData[:,:,:,:]
        print('  rgbData:', type(rgbData), rgbData.shape, rgbData.dtype)

        totalNumSlices += _slices

        if rgbDataFinal is None:
            rgbDataFinal = rgbData
        else:
            rgbDataFinal = np.concatenate((rgbDataFinal,rgbData), axis=0)
            print('  ', file, 'after concat rgbDataFinal:', type(rgbDataFinal), rgbDataFinal.shape, rgbDataFinal.dtype)

        # increment
        fileIdx += 1

    folderName = os.path.split(path)[1]
    fileName = folderName + '-rgb-merged.tif'
    savePath = os.path.join(path, fileName)
    print('  saving merged rgb to savePath:', savePath)
    print('    rgbDataFinal:', rgbDataFinal.shape)
    print('    totalNumSlices:', totalNumSlices)
    tifffile.imwrite(savePath, rgbDataFinal, imagej=True)


def run():
    import canvas

    path1 = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/FST'
    path2 = '/media/cudmore/data/Dropbox/data/whistler/data-oct-10/Morphine'
    
    pathList = [path1, path2]
    
    dictList = []
    
    for path in pathList:
        fileList = os.listdir(path)
        
        for file in fileList:
            if not file.endswith('.tif'):
                continue
        
            filePath = os.path.join(path, file)

            # load tif header
            oneHeader = canvas.canvasStackHeader(filePath)

            oneDict = {}
            
            oneDict['file'] = file

            _parentFolder = os.path.split(path)[1]
            oneDict['parentFolder'] = _parentFolder

            oneDict['xpixels'] = oneHeader['xpixels']
            oneDict['ypixels'] = oneHeader['ypixels']
            oneDict['numimages'] = oneHeader['numimages']

            oneDict['xvoxel'] = oneHeader['xvoxel']
            oneDict['yvoxel'] = oneHeader['yvoxel']
            oneDict['zvoxel'] = oneHeader['zvoxel']

            dictList.append(oneDict)

            # load load raw stack data to append stacks together
            # do this in Fiji (we need to preserve the header)

    # make df from dict List
    df = pd.DataFrame(dictList)
    print(df)

if __name__ == '__main__':
    #run()

    #loadMask()

    # load npy and save a mask .tif
    #plotCellpose()  #

    # given folder of tif, make one big rgb tif by concatenating each file
    # napari does not open this as rgb? But cellpose code in pthon is ok with it?
    #makeRgbStack()

    #postAnalysis()

    # load 3d 2-channel, calculate, and save mask
    makeMyMask()