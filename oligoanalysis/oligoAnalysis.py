"""
20221031
"""
import os
from pprint import pprint
import json

import numpy as np
import pandas as pd

import tiffile

from skimage.transform import resize  # rescale, downscale_local_mean
#from skimage.filters import gaussian
from skimage.filters import threshold_otsu

from scipy.ndimage import zoom  # to redice x/y size of image
import scipy.ndimage

from aicsimageio import AICSImage

from oligoanalysis.loadCzi import loadFolder, loadCziHeader
from oligoanalysis._logger import logger

#from oligoUtil import getEightBit, printStack
import oligoanalysis.oligoUtils

class oligoAnalysis():
    def __init__(self, path : str, xyScaleFactor : float = 0.25):
        """
        Args:
            path: Full path to czi file
            xyScaleFactor: fraction to zoom x/y
                cellpose wants nuclei to be ~10 pixels but our are ~30 pixels
        """
        logger.info(f'path: {path} xyScaleFactor:{xyScaleFactor}')
        
        if not os.path.isfile(path):
            raise ValueError(f'Did not find file: {path}')
        
        self._path : str = path

        # self._xyScaleFactor : float = xyScaleFactor
        # self._gaussianSigma = 1  # could be (0, 1, 1)

        self._header : dict = loadCziHeader(path)
        self._header['xyScaleFactor'] = xyScaleFactor
        #
        self._header['gaussianSigma'] = 1
        self._header['otsuThreshold'] = None
        self._header['numStackPixels'] = None
        self._header['numMaskPixels'] = None
        self._header['maskPercent'] = None
        #
        self._header['erodeIterations'] = 1
        self._header['dilateIterations'] = 1
        
        self.loadHeader()

        self._dfLabels : pd.DataFrame = self.loadLabelDf()
        # Created in analizeOligoDapi()

        # TODO: implement load() and save()
        # these should all start as None until load
        # this way we can provide _header and _dfLabels
        # to a folder of oligoAnalysis
        
        self._rgbStack = None
        self._cellPoseMask = None
        self._redImageMask = None
        self._redImageFiltered = None
        self._dapiFinalMask = None

        self._isLoaded = False

        '''
        self._rgbStack = self._getRgbStack()
        """rgb stack we use to run a trained cellpose model on
        """

        self._cellPoseMask = self.getCellPoseMask()
        # Output of cellpose in _seg.npy

        # _dict, self._redImageMask = self.makeImageMask()
        self._redImageMask = self.getImageMask('red')

        # analyze with ring
        self._dapiFinalMask = self.loadDapiFinalMask()
        if self._dapiFinalMask is None:
            self._dapiFinalMask = self.analyzeOligoDapi()
        '''

    def isLoaded(self):
        return self._isLoaded
    
    def load(self):
        self._isLoaded = True
        
        self._rgbStack = self._getRgbStack()
        """rgb stack we use to run a trained cellpose model on
        """

        self._cellPoseMask = self.getCellPoseMask()
        # Output of cellpose in _seg.npy

        # _dict, self._redImageMask = self.makeImageMask()
        self._redImageMask = self.loadImageMask('red')
        if self._redImageMask is None:
            self.analyzeImageMask('red')

        self._redImageFiltered = self.loadImageFiltered('red')
        if self._redImageFiltered is None:
            self.analyzeImageMask('red')

        # analyze with ring
        self._dapiFinalMask = self.loadDapiFinalMask()
        if self._dapiFinalMask is None:
            self._dapiFinalMask = self.analyzeOligoDapi()

    def save(self):
        logger.info(f'Saving analysis: {self.filename}')
        self.saveHeader()
        self.saveLabelDf()

        if self._redImageMask is not None:
            maskPath = self.getImageMaskPath('red')
            tiffile.imsave(maskPath, self._redImageMask)

        if self._redImageFiltered is not None:
            filteredPath = self.getImageFilteredPath('red')
            tiffile.imsave(filteredPath, self._redImageFiltered)

        self.saveDapiFinalMask()

    @property
    def filename(self) -> str:
        """Get the original filename.
        """
        return os.path.split(self._path)[1]
    
    def getHeader(self):
        return self._header

    def saveHeader(self):
        """Save header as json.
        """
        headerPath = self.getHeaderFile()
        logger.info(f'saving header json: {headerPath}')
        with open(headerPath, 'w') as f:
            json.dump(self._header, f, indent=4)

    def loadHeader(self):
        """Save header as json.
        """
        headerPath = self.getHeaderFile()
        if not os.path.isfile(headerPath):
            return
        logger.info(f'loading header json: {headerPath}')
        with open(headerPath, 'r') as f:
            self._header = json.load(f)

    def getHeaderFile(self):
        """Get path to save/load head.
        
        We want to save header when we change analysis params
            - gaussianSigma
            - dilation/erosion iteration
            - ???
        """
        _rgbPath = self.getRgbPath()
        headerFile = os.path.split(_rgbPath)[1]
        headerFile, _ext = os.path.splitext(headerFile)
        headerFile += f'-header.json'
        headerPath = os.path.join(self.getSaveFolder(), headerFile)
        return headerPath

    def loadDapiFinalMask(self):
        """Load _dapiFinalMask
        """
        baseFilePath = self.getBaseSaveFile()
        baseFilePath += '-dapi-final-mask.tif'
        
        if not os.path.isfile(baseFilePath):
            logger.info(f'did not find: {baseFilePath}')
            return
        logger.info(f'loading dapi_final_mask: {baseFilePath}')
        dapiFinalMask = tiffile.imread(baseFilePath)
        return dapiFinalMask

    #def saveDapiFinalMask(self, dapi_final_mask : np.ndarray = None):
    def saveDapiFinalMask(self):
        """Save _dapiFinalMask
        """
        if self._dapiFinalMask is None:
            return
        baseFilePath = self.getBaseSaveFile()
        baseFilePath += '-dapi-final-mask.tif'
        logger.info(f'saving dapi_final_mask: {baseFilePath}')
        tiffile.imsave(baseFilePath, self._dapiFinalMask)
        #tiffile.imsave(baseFilePath, dapi_final_mask)

    def saveLabelDf(self):
        dfPath = self.getLabelFile()
        if self._dfLabels is not None:
            logger.info(f'saving label df: {dfPath}')
            self._dfLabels.to_csv(dfPath, index=False)

    def loadLabelDf(self):
        dfPath = self.getLabelFile()
        if not os.path.isfile(dfPath):
            return
        logger.info(f'loading label df: {dfPath}')
        return pd.read_csv(dfPath)

    def getLabelFile(self):
        """Get path to save/load label df (self._dfMaster).
        
        See: analizeOligoDapi()
        """
        _rgbPath = self.getRgbPath()
        dfFile = os.path.split(_rgbPath)[1]
        dfFile, _ext = os.path.splitext(dfFile)
        dfFile += f'-labels.csv'
        dfPAth = os.path.join(self.getSaveFolder(), dfFile)
        return dfPAth

    def getSaveFolder(self) -> str:
        """Get the save folder and make if neccessary.
        
        This is <parent>/<parent>-analysis
        """
        _folder, _file = os.path.split(self._path)
        _, _parentFolder = os.path.split(_folder)
        
        # _saveFolder is shared by all files in parentFolder
        _saveFolder = os.path.join(_folder, _parentFolder + '-analysis')
        if not os.path.isdir(_saveFolder):
            logger.info(f'making analysis save folder: {_saveFolder}')
            os.mkdir(_saveFolder)
        
        # each raw tif/czi go into a different folder
        _cellFolder = os.path.splitext(_file)[0]
        _cellFolder = os.path.join(_saveFolder, _cellFolder)
        if not os.path.isdir(_cellFolder):
            logger.info(f'making analysis folder for file "{_file}": {_cellFolder}')
            os.mkdir(_cellFolder)

        return _cellFolder

    def getBaseSaveFile(self):
        saveFile = os.path.split(self._path)[1]
        saveFile, _ext = os.path.splitext(saveFile)
        baseSavePath = os.path.join(self.getSaveFolder(), saveFile)
        return baseSavePath

    def getRgbPath(self) -> str:
        """Get full path to saved rgb tif.
        """
        rgbSaveFile = os.path.split(self._path)[1]
        rgbSaveFile, _ext = os.path.splitext(rgbSaveFile)
        rgbSaveFile += '-rgb-small.tif'
        rgbSavePath = os.path.join(self.getSaveFolder(), rgbSaveFile)
        return rgbSavePath

    def getImageMaskPath(self, channelStr : str) -> str:
        _rgbPath = self.getRgbPath()
        maskFile = os.path.split(_rgbPath)[1]
        maskFile, _ext = os.path.splitext(maskFile)
        maskFile += f'-mask-{channelStr}.tif'
        maskPath = os.path.join(self.getSaveFolder(), maskFile)
        return maskPath

    def getImageFilteredPath(self, channelStr : str) -> str:
        _rgbPath = self.getRgbPath()
        maskFile = os.path.split(_rgbPath)[1]
        maskFile, _ext = os.path.splitext(maskFile)
        maskFile += f'-filtered-{channelStr}.tif'
        maskPath = os.path.join(self.getSaveFolder(), maskFile)
        return maskPath

    def _getRedChannel(self):
        return self._rgbStack[:, :, :, 0]

    def _getGreenChannel(self):
        return self._rgbStack[:, :, :, 1]

    def _getRgbStack(self):
        """Load or make an rgb stack from raw file.
        
        If tif exists then load, otherwise make and save.
        """
        rgbSavePath = self.getRgbPath()

        if os.path.isfile(rgbSavePath):
            logger.info(f'Loading rgb stack: {rgbSavePath}')
            _rgbStack =  tiffile.imread(rgbSavePath)
        else:
            img = AICSImage(self._path)
            imgData = img.get_image_data("ZYXC", T=0)
            # imgData is like: (21, 784, 784, 2)
            logger.info(f'loaded imgData: {imgData.shape}')

            # convert to 8 bit, we need to do each channel to maximize histogram
            imgData_ch1 = imgData[:,:,:,0]
            imgData_ch2 = imgData[:,:,:,1]

            imgData_ch1 = oligoUtils.getEightBit(imgData_ch1)
            imgData_ch2 = oligoUtils.getEightBit(imgData_ch2)

            oligoAnalysis.oligoUtils.printStack(imgData_ch1, '8-bit imgData_ch1')
            oligoAnalysis.oligoUtils.printStack(imgData_ch2, '8-bit imgData_ch2')
            
            # make rgb stack, assuming we loaded 'ZYXC'
            _shape = imgData.shape
            _rgbDim = (_shape[0], _shape[1], _shape[2], imgData.shape[3]+1)
            _rgbStack = np.ndarray(_rgbDim, dtype=np.uint8)
            _rgbStack[:,:,:,0] = imgData_ch1
            _rgbStack[:,:,:,1] = imgData_ch2
            _rgbStack[:,:,:,2] = 0

            # for Whistler, we need to make image 1/4 size
            # cellpose want nuclei to be about 10 pixels
            # Whistler data is zoomed in and nuclei are like 30 pixels
            xyScaleFactor = self._header['xyScaleFactor']
            _zoom = (1, xyScaleFactor, xyScaleFactor, 1)
            _rgbStack = zoom(_rgbStack, _zoom)
            oligoUtils.printStack(_rgbStack, 'after zoom _rgbStack')

            logger.info(f'  saving rgb stack: {rgbSavePath}')
            tiffile.imwrite(rgbSavePath, _rgbStack)

        return _rgbStack

    def getCellPoseMask(self) -> np.ndarray:
        """Get the cellpose mask from _seg.npy file.

        This is saved by cellpose outside oligoAnalysis
        """
        _rgbPath = self.getRgbPath()
        _path, _file = os.path.split(_rgbPath)
        cellPoseFile = os.path.splitext(_file)[0] + '_seg.npy'
        cellPoseSegPath = os.path.join(_path, cellPoseFile)
        if not os.path.isfile(cellPoseSegPath):
            logger.warning(f'Did not find cellpose _seg.npy file:')
            logger.warning(f'    {cellPoseSegPath}')
            return
        dat = np.load(cellPoseSegPath, allow_pickle=True).item()
        masks = dat['masks']
        
        #self._cellPoseMask = masks
        return masks

    def getImageMask(self, channelStr : str):
        if channelStr=='red':
            return self._redImageMask

    def getImageFiltered(self, channelStr : str):
        if channelStr=='red':
            return self._redImageFiltered

    def loadImageMask(self, channelStr : str):
        """Load the filtered thresholded binary mask.
        
        Created in analyzeImageMask()
        """
        maskPath = self.getImageMaskPath(channelStr)
        if os.path.isfile(maskPath):
            # load
            logger.info(f'Loading image mask "{self.filename}" {channelStr} {maskPath}')
            imgData_binary = tiffile.imread(maskPath)
            return imgData_binary

    def loadImageFiltered(self, channelStr : str):
        """Load the filtered thresholded binary mask.
        
        Created in analyzeImageMask()
        """
        maskPath = self.getImageFilteredPath(channelStr)
        if os.path.isfile(maskPath):
            # load
            logger.info(f'Loading image filtered "{self.filename}" {channelStr} {maskPath}')
            imgData_binary = tiffile.imread(maskPath)
            return imgData_binary

    def analyzeImageMask(self, channelStr : str, forceMake : bool = False):
        """Get a binary image mask.
            - Gaussian blur
            - Otsu threshold

        Args:
            channelStr: in ['red', 'green']
            forceMake: IF True then do not load, always remake
        
        Assigns:
            self._redImageMask
        """
        
        # maskPath = self.getImageMaskPath(channelStr)
        # if not forceMake and os.path.isfile(maskPath):
        #     # load
        #     logger.info(f'Loading image mask: {maskPath}')
        #     imgData_binary = tiffile.imread(maskPath)
        #     return imgData_binary

        if channelStr=='red':
            imgData = self._getRedChannel()
        elif channelStr=='green':
            imgData = self._getGreenChannel()

        _gaussianSigma = self._header['gaussianSigma']
        
        logger.info(f'{self.filename} channelStr:{channelStr} forceMake:{forceMake} _gaussianSigma:{_gaussianSigma}')
        
        otsuThreshold, imgData_blurred, imgData_binary = \
            oligoanalysis.oligoUtils.getOtsuThreshold(imgData, sigma=_gaussianSigma)
        
        # calculate pixel stats
        numStackPixels = imgData_binary.size
        numMaskPixels = np.count_nonzero(imgData_binary)
        maskPercent = numMaskPixels / numStackPixels * 100
        
        logger.info(f'  -- RESULTS: otsuThreshold:{otsuThreshold} maskPercent:{maskPercent}')

        #self._header['gaussianSigma'] = numStackPixels
        self._header['otsuThreshold'] = otsuThreshold
        self._header['numStackPixels'] = numStackPixels
        self._header['numMaskPixels'] = numMaskPixels
        self._header['maskPercent'] = maskPercent

        #
        # save
        # self.saveHeader()
        # logger.info(f'saving mask: {maskPath}')
        # tiffile.imsave(maskPath, imgData_binary)

        if channelStr=='red':
            self._redImageMask = imgData_binary
            self._redImageFiltered = imgData_blurred
        elif channelStr=='green':
            logger.warning(f'not implemented for channelStr {channelStr}')

        return imgData_binary, imgData_blurred

    def analyzeOligoDapi(self, dilateIterations : int = None,
                        erodeIterations : int = None):
        """
        Requires:
            Cellpose dapi mask
            
        Returns:
            dapi_final_mask
        """

        # this is the dapi mask output by cellpose
        # it will not exist if we did not run cllpose on this stack
        logger.info(f'{self.filename}')
        
        _cellPoseDapiMask = self.getCellPoseMask()
        if _cellPoseDapiMask is None:
            logger.warning('Did not perform ring analysis, no cellpose dapi mask')
            return

        if dilateIterations is None:
            dilateIterations = self._header['dilateIterations']
        else:
            self._header['dilateIterations'] = dilateIterations
        if erodeIterations is None:
            erodeIterations = self._header['erodeIterations']
        else:
            self._header['erodeIterations'] = erodeIterations
                
        # TODO: sloppy, we don't always need to save
        self.saveHeader()

        maskLabelList = np.unique(_cellPoseDapiMask)

        dapi_final_mask = np.zeros_like(_cellPoseDapiMask)  # dapi mask after dilation
        logger.info(f'making dapi_dilated_mask: {dapi_final_mask.shape} {dapi_final_mask.dtype}')

        listOfDict = []  # convert to pandas dataframe at end

        for maskLabel in maskLabelList:
            if maskLabel == 0:
                # background
                continue
            
            _oneMask = _cellPoseDapiMask == maskLabel  # (46, 196, 196)
            #print('_oneMask:', type(_oneMask), _oneMask.shape, _oneMask.dtype)

            # dilate the mask
            if dilateIterations>0:
                _dilatedMask = scipy.ndimage.binary_dilation(_oneMask, iterations=dilateIterations)
            else:
                _dilatedMask = _oneMask

            if erodeIterations>0:
                _erodedMask = scipy.ndimage.binary_erosion(_oneMask, iterations=erodeIterations)
            else:
                _erodedMask = _oneMask

            #print('  dilatedMask:', type(dilatedMask), dilatedMask.shape, dilatedMask.dtype, np.sum(dilatedMask))

            # make a ring
            #dilatedMask = dilatedMask ^ _oneMask
            finalMask = _dilatedMask ^ _erodedMask  # carrot (^) is xor

            # the number of pixels in the dilated/eroded dapi mask
            finalMaskCount = np.count_nonzero(finalMask)

            # oligo red mask pixels in the (dilated/eroded) dapi mask
            redImageMask = np.where(finalMask==True, self._redImageMask, 0)  # 0 is fill value
            #print('  redImageMask:', type(redImageMask), redImageMask.shape, redImageMask.dtype, np.sum(redImageMask))

            # like cellpose_dapi_mask but after dilation
            finalMaskLabel = finalMask.copy().astype(np.int64)
            #print('1 ', dilatedMaskLabel.dtype, np.max(dilatedMaskLabel))
            # +1 so colors are different from cellpose_dapi_mask
            finalMaskLabel[finalMaskLabel>0] = maskLabel + 1   
            #print('  2 ', dilatedMaskLabel.dtype, np.max(dilatedMaskLabel))
            dapi_final_mask = dapi_final_mask + finalMaskLabel
            #print('  dapi_dilated_mask:', dapi_dilated_mask.shape, np.sum(dapi_dilated_mask))
            
            redImageMaskPercent = np.sum(redImageMask) / finalMaskCount * 100
            
            oneDict = {
                'label': maskLabel,
                'finalMaskCount': finalMaskCount,  # num pixels in dilated mask
                'redImageMaskSum': np.sum(redImageMask),  # sum of red mask in dilated dapi mask
                'redImageMaskPercent': redImageMaskPercent,  # fraction of pixels in red mask in dilated mask
            }
            listOfDict.append(oneDict)

            # no, labelIdx corresponds to napari label
            #    NO: labelIdx here is (label + 1) in napari
            #    NO: labelIdx 52 corresponds to 53 in napari
            # in napari, label (83, 88) are 100% positive dapi+oligo
            
        self._dfLabels = pd.DataFrame(listOfDict)
        
        # self.saveLabelDf()

        # save ring, dapi_final_mask
        #self._dapiFinalMask = dapi_final_mask
        #self.saveDapiFinalMask(dapi_final_mask)

        # these are the dapi masks with a good amount of red
        redImageMaskPercent_threshold = 5
        print('\nredImageMaskPercent_threshold:', redImageMaskPercent_threshold)
        print('Using this threshold, we have the following DAPI mask candidates')
        print('. e.g. the ones with a lot of Oligo red)')
        print(self._dfLabels[self._dfLabels['redImageMaskPercent']>=redImageMaskPercent_threshold])

        return dapi_final_mask

def check_OligoAnalysis():
    cziPath = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/FST/B35_Slice2_RS_DS1.czi'
    oa = oligoAnalysis(cziPath)

    oa.analyzeOligoDapi()

    oa.analyzeOligoDapi()

if __name__ == '__main__':
    #check_OligoAnalysisFolder()

    check_OligoAnalysis()