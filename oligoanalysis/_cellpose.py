"""
Run cellpose on 3d rgb stacks and save _seg.npy file
"""
import os
import pathlib
import tifffile

import numpy as np

from cellpose import models
from cellpose import utils, io

from cellpose.io import logger_setup

from oligoanalysis._logger import logger

def getCellposeLog():
    """Get the full path to the <user> cellpose log file.
    """
    userPath = pathlib.Path.home()
    userCellposeFolder = os.path.join(userPath, '.cellpose')
    if not os.path.isdir(userCellposeFolder):
        logger.error(f'Did not find <user>.cellpose "{userCellposeFolder}"')
        return None
    
    _runLogFile = os.path.join(userCellposeFolder, 'run.log')
    if not os.path.isfile(_runLogFile):
        logger.error(f'Did not find user cellpose log file: "{_runLogFile}"')
        return None

    return _runLogFile

def getModels():
    """Return str list of full path for all models in 'oligoanalysis/models' folder.
    """

    import oligoanalysis
    _filepath = os.path.abspath(oligoanalysis.__file__)
    _folder, _ = os.path.split(_filepath)
    _folder = os.path.join(_folder, 'models')
    
    #logger.info(_folder)
    
    if not os.path.isdir(_folder):
        logger.error(f'did not find model path: {_folder}')
        return

    files = os.listdir(_folder)
    fileList = [os.path.join(_folder, file)
                    for file in sorted(files)]
    return fileList

def runModelOnImage(imgPath : str, imgData : np.ndarray = None, setupLogger=True):
    """Run our pre-defined (trained) model on one 3D RGB stack.
    
    Args:
        imgPath: full path to 3D rgb stack
        imgData: image data for 3D rgb stack
        setupLogger: if True will re-init logger in <user>/.cellpose/run.log
    """
    
    # setup cellpose logging to file
    #  <user>/.cellpose/run.log (see getCellposeLog())
    if setupLogger:
        logger_setup()
    
    if imgData is None:
        imgData = tifffile.imread(imgPath)

    logger.info('cellpose runModelOnImage()')
    logger.info(f'  {imgPath}')  # (7, 196, 196, 3)
    logger.info(f'  {imgData.shape}')  # (7, 196, 196, 3)

    gpu = False
    model_type = None  # 'cyto' or 'nuclei' or 'cyto2'
    
    pretrained_models = getModels()
    if pretrained_models is None:
        logger.warning('Did not find any models, cellpose is not running')
        return

    pretrained_model= pretrained_models[1]  # '/Users/cudmore/Sites/oligo-analysis/models/CP_20221008_110626'
    
    channels = [[2,1]]  # # grayscale=0, R=1, G=2, B=3
    diameter = 16.0
    do_3D = True
    net_avg = False

    # run model
    # masks, flows, styles, diams = model.eval(imgData,
    #                                 diameter=diameter, channels=channels,
    #                                 do_3D=do_3D)

    logger.info(f'  instantiating model with models.CellposeModel()')
    logger.info(f'    gpu: {gpu}')
    logger.info(f'    model_type: {model_type}')
    logger.info(f'    pretrained_model: {pretrained_model}')
    logger.info(f'    net_avg: {net_avg}')

    model = models.CellposeModel(gpu=gpu, model_type=model_type,
                                    pretrained_model=pretrained_model,
                                    net_avg=net_avg)

    logger.info('  running model.eval')
    logger.info(f'    diameter: {diameter}')
    logger.info(f'    channels: {channels}')
    logger.info(f'    do_3D: {do_3D}')

    masks, flows, styles = model.eval(imgData,
                                        diameter=diameter, channels=channels,
                                        do_3D=do_3D
                                        )

    # save
    logger.info(f'  saving cellpose _seg.npy into folder {os.path.split(imgPath)[0]}')
    # models.CellposeModel.eval does not return 'diams', using diameter
    io.masks_flows_to_seg(imgData, masks, flows, diameter, imgPath, channels)
    
    logger.info(f'  >>> DONE running cellpose on pre-trained model "{os.path.split(pretrained_model)[1]}" for file "{os.path.split(imgPath)[1]}"')

    # can't save 3d output as png
    # io.save_to_png(imgData, masks, flows, imgPath)

if __name__ == '__main__':
    path = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/FST/FST-analysis/B35_Slice2_RS_DS1.czi/B35_Slice2_RS_DS1-rgb-small.tif'
    runModelOnImage(imgPath=path)

    # modelList = getModels()
    # for model in modelList:
    #     print(model)

    # logFile = getCellposeLog()
    # print(logFile)