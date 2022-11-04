"""
20221101
"""
import time
from typing import List, Union  # , Callable, Iterator, Optional

import numpy as np

#from skimage.measure import regionprops, regionprops_table
import skimage.measure

from qtpy import QtWidgets, QtCore, QtGui

import napari
import napari_layer_table  # our custom plugin (installed from source)

#import magicgui.widgets

import oligoanalysis.oligoUtils
from oligoanalysis import oligoAnalysis
from oligoanalysis import oligoAnalysisFolder

from oligoanalysis.interface import myTableView
from oligoanalysis.interface._data_model import pandasModel

from oligoanalysis.interface import bHistogramWidget

from oligoanalysis._logger import logger

def old_runPlugin(viewer : napari.Viewer, layer, onAddCallback=None):
    """Run the napari layer table plugin on one layer.
    
    Args:
        viewer: A napari viewer
        layer:
        onAddCallback:
        """
    # create the plugin
    # ltp is type: napari_layer_table._my_widget.LayerTablePlugin
    ltp = napari_layer_table.LayerTablePlugin(viewer, oneLayer=layer, onAddCallback=onAddCallback)
    #ltp.signalDataChanged.connect(on_user_edit_points2)
    
    # show
    area = 'right'
    name = layer.name
    _dockWidget = viewer.window.add_dock_widget(ltp, area=area, name=name)
    
    # remove
    #viewer.window.remove_dock_widget(_dockWidget) 
    
    return ltp

class oligoInterface(QtWidgets.QWidget):

    signalSelectImageLayer = QtCore.Signal(object, object, object)
    """Emit when user changes to image layer in napari viewer.
    
    Args:
        data (np.ndarray) image data
        name (str) name of the layer
        colorMapName (Str) name of color map, like 'red' or 'green'
    """

    signalSetSlice = QtCore.Signal(object)
    """Emit when user changes slice slider in napari viewer.
    
    Args:
        sliceNumber (int)
    """

    def __init__(self, viewer : napari.Viewer, folderPath : str, parent = None):
        super().__init__(parent)
        self._viewer = viewer

        self._folderPath = folderPath

        self._oligoAnalysisFolder = oligoAnalysisFolder(folderPath)

        self._layerTableDocWidget = None
        
        self._selectedFile : str = None
        self._selectedRow : int = None
        # name of the selected file

        self._buildGui()
        #self.refreshAnalysisTable()

        self._buildingNapari = False
        # to pause updates, set True when adding/removing viewer layers

        # respond to change in layer
        self._viewer.layers.selection.events.changed.connect(self.slot_selectLayer)

        # respond to changes in image slice
        self._viewer.dims.events.current_step.connect(self.slot_setSlice)

        self.updateStatus('Ready')

    def slot_setSlice(self, event):
        """Respond to change in image slice slider in viewer.
        """
        if self._buildingNapari:
            return
        logger.info(f'event.type: {event.type}')
        logger.info(f'event: {type(event)}')
        print('    ', self._viewer.dims.current_step)

        # query the global viewer (I don't like this)
        current_step_tuple = self._viewer.dims.current_step  # return tuple (slice, ?, ?)
        currentSlice = current_step_tuple[0]
        self.signalSetSlice.emit(currentSlice)

    def slot_selectLayer(self, event):
        """Respond to change in layer selection in viewer.
        
        Args:
            event (napari.utils.events.event.Event): event.type == 'changed'

        Notes:
            We receive this event multiple times, we want all info in `event`
                but not sure how to query it?
            For now, we are using the global self._viewer
        """
        if self._buildingNapari:
            return

        # _activeLayer will sometimes be None
        _activeLayer = self._viewer.layers.selection.active
        if _activeLayer is None:
            return
        # napari.layers.image.image.Image
        if isinstance(_activeLayer, napari.layers.image.image.Image):
            # logger.info(f'event.type: {event.type}')
            # print('  ', type(event))
            # print('  _activeLayer:', _activeLayer)
            # print('  _activeLayer:', type(_activeLayer))
            
            _data = _activeLayer.data
            _name = _activeLayer.name
            _colorMapName = _activeLayer.colormap.name

            # napari.utils.Colormap: colormap
            #print(_activeLayer.colormap)  # 2d array of [r,g,b,a]
            print(f'  _activeLayer.colormap.name:{_activeLayer.colormap.name}')

            # TODO: when our hist signals contrast slider change,
            #   change layer.events.contrast_limits
            logger.info(f'  -->> emit signalSelectImageLayer _data.shape {_data.shape} _name:{_name}')
            self.signalSelectImageLayer.emit(_data, _name, _colorMapName)

    def _buildTableView(self):
        #  table/list view
        _myTableView = myTableView()
        
        # TODO (Cudmore) Figure out how to set font of (cell, row/vert header, col/horz header)
        #   and reduce row size to match font
        _fontSize = 11
        aFont = QtGui.QFont('Arial', _fontSize)
        _myTableView.setFont(aFont)  # set the font of the cells
        _myTableView.horizontalHeader().setFont(aFont)
        _myTableView.verticalHeader().setFont(aFont)

        _myTableView.verticalHeader().setDefaultSectionSize(_fontSize)  # rows
        _myTableView.verticalHeader().setMaximumSectionSize(_fontSize)
        #_myTableView.horizontalHeader().setDefaultSectionSize(_fontSize)  # rows
        #_myTableView.horizontalHeader().setMaximumSectionSize(_fontSize)
        _myTableView.resizeRowsToContents()

        _myTableView.signalSelectionChanged.connect(self.on_table_selection)

        # this is a full refresh of table
        dfAnalysis = self._oligoAnalysisFolder.getAnalysisDataFrame(removeColumns=True)
        myModel = pandasModel(dfAnalysis)
        _myTableView.mySetModel(myModel)

        return _myTableView

    def _buildGui(self):
        vLayout = QtWidgets.QVBoxLayout()

        # top row of controls
        hLayout = QtWidgets.QHBoxLayout()

        aButton = QtWidgets.QPushButton('Load Folder')
        aButton.clicked.connect(self.on_load_folder_button)
        hLayout.addWidget(aButton)

        aButton = QtWidgets.QPushButton('Save')
        aButton.clicked.connect(self.on_save_button)
        hLayout.addWidget(aButton)

        # checkboxes to toggle interface
        aCheckbox = QtWidgets.QCheckBox('Histogram')
        aCheckbox.setChecked(True)
        aCheckbox.stateChanged.connect(self.on_histogram_checkbox)
        hLayout.addWidget(aCheckbox)

        aCheckbox = QtWidgets.QCheckBox('Points')
        aCheckbox.setChecked(True)
        aCheckbox.stateChanged.connect(self.on_points_checkbox)
        hLayout.addWidget(aCheckbox)

        vLayout.addLayout(hLayout)

        # table of results, update this as we analyze
        #dfAnalysis = self._oligoAnalysisFolder.getAnalysisDataFrame(removeColumns=True)
        # self._analysisTable = magicgui.widgets.Table(dfAnalysis)
        # self._analysisTable.read_only = True
        # self._analysisTable.native.itemClicked.connect(self.on_table_selection)
        # self._aTable.native: magicgui.backends._qtpy.widgets._QTableExtended
        # vLayout.addWidget(self._analysisTable.native)
        self._analysisTable = self._buildTableView()
        vLayout.addWidget(self._analysisTable)

        # table of files
        # self._aTable = magicgui.widgets.Table(self._oligoAnalysisFolder.getDataFrame())
        # self._aTable.read_only = True
        # self._aTable.native.itemClicked.connect(self.on_table_selection)
        # # self._aTable.native: magicgui.backends._qtpy.widgets._QTableExtended
        # vLayout.addWidget(self._aTable.native)

        # red mask
        hLayout = QtWidgets.QHBoxLayout()
        aButton = QtWidgets.QPushButton('Make Red Mask')
        aButton.clicked.connect(self.on_make_red_mask)
        hLayout.addWidget(aButton)

        self._sigmaLineWidget = QtWidgets.QLineEdit('1')
        self._sigmaLineWidget.editingFinished.connect(self.on_edit_sigma)
        hLayout.addWidget(self._sigmaLineWidget)

        vLayout.addLayout(hLayout)

        # ring mask
        hLayout = QtWidgets.QHBoxLayout()
        aButton = QtWidgets.QPushButton('Make Ring Mask')
        aButton.clicked.connect(self.on_make_ring_mask)
        hLayout.addWidget(aButton)

        aLabel = QtWidgets.QLabel('Erode')
        hLayout.addWidget(aLabel)
        self._erodeSpinBox = QtWidgets.QSpinBox()
        self._erodeSpinBox.setMinimum(0)
        self._erodeSpinBox.setValue(1)
        # self._erodeSpinBox.valueChanged.connect(self.on_edit_erode_dilate)
        hLayout.addWidget(self._erodeSpinBox)

        aLabel = QtWidgets.QLabel('Dilate')
        hLayout.addWidget(aLabel)
        self._dilateSpinBox = QtWidgets.QSpinBox()
        self._dilateSpinBox.setMinimum(0)
        self._dilateSpinBox.setValue(1)
        # self._dilateSpinBox.valueChanged.connect(self.on_edit_erode_dilate)
        hLayout.addWidget(self._dilateSpinBox)

        vLayout.addLayout(hLayout)

        #
        self._statusWidget = QtWidgets.QLabel('Status')
        vLayout.addWidget(self._statusWidget)

        # need pointer to set _imgData on switching to an image layer
        _empty = np.zeros((1,1,1))
        self._bHistogramWidget = bHistogramWidget(_empty)
        self.signalSelectImageLayer.connect(self._bHistogramWidget.slot_setData)
        self.signalSetSlice.connect(self._bHistogramWidget.slot_setSlice)
        vLayout.addWidget(self._bHistogramWidget)

        #
        self.setLayout(vLayout)

    def updateStatus(self, text : str):
        text = f'Status: {text}'
        self._statusWidget.setText(text)

    def refreshAnalysisTable(self):
        dfAnalysis = self._oligoAnalysisFolder.getAnalysisDataFrame(removeColumns=True)
        #self._analysisTable.value = dfAnalysis
        
        # want to just set one row
        # self._analysisTable.myModel.mySetRow(rowList, df)
        # mySetRow(self, rowList: List[int], df: pd.DataFrame):

        # self._selectedFile
        # self._selectedRow

        rowList = [self._selectedRow]
        
        df = dfAnalysis[dfAnalysis['file']==self._selectedFile]
        self._analysisTable.myModel.mySetRow(rowList, df)

        # this is a full refresh of table
        # myModel = pandasModel(dfAnalysis)
        # self._analysisTable.mySetModel(myModel)

    def on_histogram_checkbox(self, state):
        logger.info(f'state:{state}')
        checked = state > 0
        self._bHistogramWidget.setVisible(checked)

    def on_points_checkbox(self, state):
        logger.info(f'state:{state}')
        
    def on_load_folder_button(self):
        logger.info('')
    
    def on_save_button(self):
        logger.info('')
        oa = self.getSelectedAnalysis()
        if oa is None:
            return
        oa.save()

    def on_edit_sigma(self):
        text = self._sigmaLineWidget.text()
        logger.info(f'text: {text}')
        if ',' in text:
            # tuple
            #value = tuple(text)
            try:
                text = text.replace('(', '')
                text = text.replace(')', '')
                value = [float(x) for x in text.split(',')]
                value = tuple(value)
            except (ValueError) as e:
                logger.error('Plase enter a number or a list of numbers')
                return
        else:
            # int
            value = float(text)
        logger.info(f'  value:{type(value)} {value}')
        return value

    def on_make_red_mask(self):
        """Remake red mask using xxx as parameter.
        """  
        logger.info('')

        # filename = self._selectedFile
        # if filename is None:
        #     return
        # oa = self._oligoAnalysisFolder.getOligoAnalysis(filename)

        oa = self.getSelectedAnalysis()
        if oa is None:
            return

        # need to set sigma in oa._header['gaussianSigma']
        _gaussianSigma = self.on_edit_sigma()
        oa._header['gaussianSigma'] = _gaussianSigma

        # refresh interface
        logger.info(f'fetching new red mask with sigma: {_gaussianSigma}')

        _redImageMask, _redImageFiltered = oa.analyzeImageMask('red')
        self._redbinaryLayer.data = _redImageMask  # oa.getImageMask('red')
        self._redFilteredLayer.data = _redImageFiltered  # oa.getImageMask('red')

        self.refreshAnalysisTable()
        
    def on_make_ring_mask(self, value):
        erodeIterations = self._erodeSpinBox.value()
        dilateIterations = self._dilateSpinBox.value()

        logger.info('')

        # filename = self._selectedFile
        # if filename is None:
        #     return
        # oa = self._oligoAnalysisFolder.getOligoAnalysis(filename)

        oa = self.getSelectedAnalysis()
        if oa is None:
            return

        # need to set sigma in oa._header['gaussianSigma']
        # oa._header['erodeIterations'] = erodeIterations
        # oa._header['dilateIterations'] = dilateIterations

        _dapiFinalMask = oa.analyzeOligoDapi(dilateIterations=dilateIterations,
                                        erodeIterations=erodeIterations)
        if _dapiFinalMask is None:
            # happend when there is no cellpose mask from _seg.npy file
            return
        
        oa._dapiFinalMask = _dapiFinalMask

        # refresh interface
        self.refreshAnalysisTable()
        self.dapiFinalMask_layer.data = _dapiFinalMask

    def old_on_table_selection(self, item : QtWidgets.QTableWidgetItem):
        logger.info(item.row())
        row = item.row()
        rowDict = self._oligoAnalysisFolder.getRow(row)
        
        # print('  selected row dict is:')
        # print(rowDict)

        filename = rowDict['file']
        self.switchFile(filename)

    def on_table_selection(self, rowList : List[int], isAlt : bool = False):
        """Respond to user selection in table (myTableView).
        
        This is called when user selects a row(s) in underlying myTableView.

        Args:
            rowList: List of rows that were selected
            isAlt: True if keyboard Alt is down
        """

        logger.info(f'rowList:{rowList} isAlt:{isAlt}')

        oneRow = rowList[0]
        rowDict = self._oligoAnalysisFolder.getRow(oneRow)
        
        # print('  selected row dict is:')
        # print(rowDict)

        filename = rowDict['file']
        self.switchFile(filename, oneRow)

    def getSelectedAnalysis(self):
        """Get the selected oligoAnalysis.
        """
        if self._selectedFile is None:
            return
        oa = self._oligoAnalysisFolder.getOligoAnalysis(self._selectedFile)
        return oa

    def switchFile(self, filename : str, row : int):
        """load oa

        Args:
            filename:
            row:
        """
        logger.info(f'filename:{filename}')

        if filename == self._selectedFile:
            logger.info(f'  file is already selected: {self._selectedFile}')
            return
        
        self._selectedRow = row
        self._selectedFile = filename

        self._viewer.title = filename
        
        self.clearViewer()  # remove all layers
        
        oa = self._oligoAnalysisFolder.getOligoAnalysis(filename)
        self.displayOligoAnalysis(oa)

        # set dilate/erode
        self._dilateSpinBox.setValue(oa._header['dilateIterations'])
        self._erodeSpinBox.setValue(oa._header['erodeIterations'])

        # set analysis table
        # xxx

    def clearViewer(self):
        """Remove all layers from the napari viewer.
        """
        self._viewer.layers.clear()
        if self._layerTableDocWidget is not None:
            self.closeLayerTablePlugin()

    def openLayerTablePugin(self, layer):
        onAddCallback = None
        ltp = napari_layer_table.LayerTablePlugin(self._viewer, oneLayer=layer, onAddCallback=onAddCallback)
        #ltp.signalDataChanged.connect(on_user_edit_points2)
        
        # show
        area = 'right'
        name = layer.name
        self._layerTableDocWidget = viewer.window.add_dock_widget(ltp, area=area, name=name)

    def closeLayerTablePlugin(self):
        viewer.window.remove_dock_widget(self._layerTableDocWidget) 
        self._layerTableDocWidget = None

    def displayOligoAnalysis(self, oa : oligoAnalysis):
        """Display all oligo analysis images in napari viewer.
        """
        self._buildingNapari = True

        viewer = self._viewer
        
        imgRgb = oa._getRgbStack()
        imgRed = oa._getRedChannel()
        imgGreen = oa._getGreenChannel()
        imgCellposeMask = oa.getCellPoseMask()  # can be None
        imgRed_binary = oa.getImageMask('red')
        imgRed_filtered = oa.getImageFiltered('red')
        dapiFinalMask = oa._dapiFinalMask  # can be None
        
        #viewer = napari.Viewer()
        xVoxel = oa._header['xVoxel']
        yVoxel = oa._header['yVoxel']
        zVoxel = oa._header['zVoxel'] / 2  # real scale looks like crap!

        scale = (zVoxel, yVoxel, xVoxel)

        imgRedLayer = viewer.add_image(imgRed, name='imgRed', scale=scale, blending='additive')
        imgRedLayer.colormap = 'red'
        imgRedLayer.contrast_limits = (0, 100)

        imgGreenLayer = viewer.add_image(imgGreen, name='imgGreen', scale=scale, blending='additive')
        imgGreenLayer.colormap = 'green'
        imgGreenLayer.contrast_limits = (0, 150)

        # we want to be able to update this image
        self._redbinaryLayer = viewer.add_labels(imgRed_binary, name='imgRed_binary', scale=scale)

        self._redFilteredLayer = viewer.add_image(imgRed_filtered, name='imgRed_filtered', scale=scale, blending='additive')
        self._redFilteredLayer.colormap = 'red'
        #self._redFilteredLayer.contrast_limits = (0, 150)

        # we will not update this, until we add runnign a model (slow)
        if imgCellposeMask is not None:
            imgMask_layer = viewer.add_labels(imgCellposeMask, name='imgCellposeMask', scale=scale)
            
        # we want to be able to update this image
        if dapiFinalMask is None:
            dapiFinalMask = np.zeros((1,1,1), dtype=np.uint64)
        self.dapiFinalMask_layer = viewer.add_labels(dapiFinalMask, name='dapiFinalMask', scale=scale)
        
        #
        # make a pnts layer from labels
        #_cellPoseMask = oa._getCellPoseMask()  # can be none
        if imgCellposeMask is not None:
            _regionprops = skimage.measure.regionprops(imgCellposeMask)

            # add centroid to napari
            _points = [
                (s.centroid[0]*zVoxel, s.centroid[1]*yVoxel, s.centroid[2]*xVoxel) 
                for s in _regionprops]  # point[i] is a tuple of (z, y, x)
            _area = [s.area for s in _regionprops]  # point[i] is a tuple of (z, y, x)
            _label = [s.label for s in _regionprops]  # point[i] is a tuple of (z, y, x)

            properties = {
                'label': _label,
                'redImageMaskPercent': oa._dfLabels['redImageMaskPercent'],
                #'finalMaskCount': _finalMaskCount,
                'area': _area,
            }

            # add points to viewer
            label_layer_points = viewer.add_points(_points,
                                                #face_color=face_color,
                                                symbol='cross',
                                                size=5,
                                                properties=properties)

            #
            self.openLayerTablePugin(label_layer_points)
        
        # set histogram to red image layer (data and name)
        self._bHistogramWidget.slot_setData(imgRedLayer.data, imgRedLayer.name)

        self._buildingNapari = False

if __name__ == '__main__':
    # cziPath = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/FST/B35_Slice2_RS_DS1.czi'
    # oa = oligoAnalysis(cziPath)

    viewer = napari.Viewer()

    folderPath = '/Users/cudmore/Dropbox/data/whistler/data-oct-10/FST'

    oi = oligoInterface(viewer, folderPath)
    oi.show()

    #oi.displayOligoAnalysis(oa)
    
    #oi.clearViewer()

    napari.run()