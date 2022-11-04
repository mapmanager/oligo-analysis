## Goal

Process raw cvi files and make a cellpose model to predict future files.

## Preprocessing

 - In Fiji, open each czi and save as tif
 - In python, using recent Canvas repo, open tiff and make a csv database with
   - filename
   - x/y/z
   - voxel x/y/z

## Working on data in folder 'data-oct-10'

```
                          file parentFolder  xpixels  ypixels  numimages    xvoxel    yvoxel  zvoxel
0        B35_Slice2_RS_DS1.tif          FST      784      784         14  0.099502  0.099502       1
1        G19_Slice1_LS_NAc.tif          FST      784      784          8  0.099502  0.099502       1
2     B36_Slice2_LS_DS_1.0.tif          FST      784      784         42  0.099502  0.099502       1
3        G20_Slice3_LS_NAc.tif          FST      784      784          8  0.099502  0.099502       1
4        G22_Slice2_RS_DS5.tif          FST      784      784         20  0.099502  0.099502       1
5        P8_Slice2_LS_NAc1.tif     Morphine      784      784         18  0.099502  0.099502       1
6        P12_Slice2_LS_DS6.tif     Morphine      784      784         24  0.099502  0.099502       1
7  P11_Slice1_LS_NAcmedial.tif     Morphine      784      784         22  0.099502  0.099502       1
8        P10_Slice2_RS_DS1.tif     Morphine      784      784         12  0.099502  0.099502       1
9        P9_Slice 3_LS_DS1.tif     Morphine      784      784         14  0.099502  0.099502       1
```

## In Fiji, manually append each stack to next to create one big stack.

We do this to then run cellpose so we have enought ROI to train.

A lot of the whistler data only has like one DAPI+Oligo nucleus. Cell pose requires at last 5 ROI!

Made concatenated stack 'fst-concat.tif', run cellpose on this. Once we have a model, run it on the same concatenated stack for 'Morphine' to see how model does.

IMPORTANT: Need to reduce x/y by 4x, cellpose has strict requirements on cell size. In general, Whistler lab needs to image with less zoom (bigger field of view)

In Fiji, reduce x/y from 784 to 196 = 784/4

Saved as 'fst-concat-small.tif'

NOTE: We lose the x/y/z scale here !!!

Cellpose wants 2-color tif files (not individual channel files)

1) make 2 channel into composite
2) composite to rgb
3) save as Image Sequence


  Need to save this combined small stack into a folder of individual slices for cell pose

  In Fiji, 'File - Save As - Image Sequence'

## Training

Train model on a number of slices with rois
  - channel 1 is green: nuclei
  - channel 2 is red: oligo
  - use default 'cyto' model on a number of slices (saving each slice to npy segmented mask)
  - once we have a number of rois (Created mostly automatically by default 'cyto' model)
  - 'train' and new model using these rois (this will save)
  - then apply that model to full rgb stack and save final npy mask

