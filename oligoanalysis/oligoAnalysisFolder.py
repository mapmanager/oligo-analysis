"""
"""
import os

import pandas as pd

from oligoanalysis import loadCzi
from oligoanalysis.oligoAnalysis import oligoAnalysis
from oligoanalysis._logger import logger

class oligoAnalysisFolder():

    def __init__(self, folderPath : str = None):
        """
        
        Args:
            folderPath: Full path to raw scope stacks.
        """
        self._folderPath : str = folderPath
        # Full path to raw scope stack

        self._dfFolder = self._loadAllFileHeader()
        # DataFrame of files in _folderPath

        self._analysisList = {}  #[None] * len(self._dfFolder)

        # load oligo analysis headers
        files = self._dfFolder['file'].tolist()
        for file in files:
            logger.info(f'  loading file header for: {file}')
            filePath = os.path.join(self._folderPath, file)
            self._analysisList[file] = oligoAnalysis(filePath)

    def getDataFrame(self):
        """Get the dataframe of loaded file headers.
        """
        return self._dfFolder
    
    def getAnalysisDataFrame(self, removeColumns=False):
        """Get a DataFrame from all oligo analysis.
        
        Args:
            removeColumns: If True, remove some columns to make more concise
        """
        removeColumnList = ['xPixels', 'yPixels',
            'yVoxel', 'xyScaleFactor', 'numStackPixels', 'numMaskPixels']
        
        dictList = []
        for k,v in self._analysisList.items():
            oa = v
            oneDict = oa.getHeader()
            dictList.append(oneDict)
        df = pd.DataFrame(dictList)
        
        if removeColumns:
            for _col in removeColumnList:
                df = df.drop(_col, axis=1)

        # logger.info('')
        # print('  got df:')
        # print(df)
        
        return df


    def getRow(self, row : int) -> dict:
        """Get one row as dict.
        """
        df = self._dfFolder.loc[row]
        return df.to_dict()
    
    def getOligoAnalysis(self, file : str):
        """Given a raw file name, return the oligoAnalysis.
        
        Make if necc.
        """
        logger.info(f'file:{file}')
        if file in self._analysisList.keys():
            oa = self._analysisList[file]
            if not oa.isLoaded():
                oa.load()
            logger.info(f'  returning: {oa}')
            return oa
        else:
            # NOT TAKEN
            filePath = os.path.join(self._folderPath, file)
            #try:
            if 1:
                print(f'  creating oligo analysis for filePath: {filePath}')
                oneAnalysis = oligoAnalysis(filePath)
                print('    oneAnalysis:', oneAnalysis)
                self._analysisList[file] = oneAnalysis
                return self._analysisList[file]
            # except (ValueError) as e:
            #     logger.error(e)

    def _loadAllFileHeader(self):
        """Load all file headers in folder.
        """
        logger.info(f'Loading folder: {self._folderPath}')
        df = loadCzi.loadFolder(self._folderPath)
        return df

def check_OligoAnalysisFolder():
    folderPath = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/FST'

    oaf = oligoAnalysisFolder(folderPath)
    print('oaf._dfFolder')
    print(oaf._dfFolder)

    oa = oaf.getOligoAnalysis('')

if __name__ == '__main__':
    check_OligoAnalysisFolder()