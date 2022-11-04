"""
Load czi files using aicsimageio
"""

import os
from pprint import pprint
from datetime import datetime

import pandas as pd

from aicsimageio import AICSImage
from aicspylibczi import CziFile

from oligoanalysis._logger import logger

def loadFolder(folderPath):
    """Load headers for a folder of czi files.
    """
    _folder, parentFolder = os.path.split(folderPath)

    fileList = os.listdir(folderPath)
    fileList = sorted(fileList)
    headerList = []
    #sumSlices = 0
    for file in fileList:
        if not file.endswith('.czi'):
            continue
        filePath = os.path.join(folderPath, file)
        header = loadCziHeader(filePath)
        
        zPixels = header['zPixels']

        # header['firstSlice_mask'] = sumSlices
        # header['lastSlice_mask'] = sumSlices + zPixels - 1
        
        # header['parentFolder'] = parentFolder
        
        #sumSlices += zPixels
        
        headerList.append(header)

    df = pd.DataFrame(headerList)
    return df

def loadCziHeader(cziPath : str) -> dict:
    """Load header from a czi file.

    Notes:
        For Whistler datetime, I am just dropping "-07:00"
        Otherwise I get exception
        "Invalid isoformat string: '2022-09-06T11:28:35.0731032-07:00'"

        Whister czi files also have one too many decimals in fractional seconds
    """
    
    with open(cziPath) as f:
        czi = CziFile(f)

    #print('czi:', [x for x in dir(czi)])
    # print('  czi.meta:')
    # print(czi.meta)

    xpath_str = "./Metadata/Information/Document/CreationDate"
    _creationdate = czi.meta.findall(xpath_str)  #  [<Element 'CreationDate' at 0x7fa670647f40>]
    #print('_creationdate:', _creationdate)
    for creationdate in _creationdate:
        # xml.etree.ElementTree.Element
        #print('  creationdate:', creationdate)
        #print(dir(creationdate))
        _datetimeText = creationdate.text
        
        # _datetime: 2022-09-06T11:28:35.0731032-07:00
        if _datetimeText.endswith('-07:00'):
            _datetimeText = _datetimeText.replace('-07:00', '')
            _datetimeText = _datetimeText[0:-1]  # remove last fractional second

        # take apart datetime
        _date = ''
        _time = ''
        try:
            #_datetime = datetime.strptime(datetime_text, "%Y-%m-%dT%H:%M:%S%f")
            #_datetime = datetime.strptime(_datetimeText, "%Y-%m-%dT%H:%M:%S.%f%z")
            _datetime = datetime.fromisoformat(_datetimeText)
            #print(_datetime)
            _date = _datetime.strftime('%Y-%m-%d')
            _time = _datetime.strftime('%H-%M-%S')
        except (ValueError) as e:
            print(f'EXCEPTION IN PARSING TIME STR "{_datetimeText}"')
            print(e)

        img = AICSImage(cziPath)  # selects the first scene found
    
        xVoxel = img.physical_pixel_sizes.X
        yVoxel = img.physical_pixel_sizes.Y
        zVoxel = img.physical_pixel_sizes.Z

        xPixels = img.dims.X
        yPixels = img.dims.Y
        zPixels = img.dims.Z

        _parentFolder, _ = os.path.split(cziPath)
        _, _parentFolder = os.path.split(_parentFolder)
        
        _header = {
            'file': os.path.split(cziPath)[1],
            'parentFolder': _parentFolder,
            'date': _date,
            'time': _time,
            'xPixels': xPixels,
            'yPixels': yPixels,
            'zPixels': zPixels,
            'xVoxel': xVoxel,
            'yVoxel': yVoxel,
            'zVoxel': zVoxel,
        }

        return _header

if __name__ == '__main__':

    # cziPath = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/FST/B35_Slice2_RS_DS1.czi'
    # header = loadCzi(cziPath)
    # pprint(header)

    folderPath = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/FST'
    dfFolder = loadFolder(folderPath)

    print(dfFolder)