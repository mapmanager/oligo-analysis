
DAPI/Oligo analysis for the Whistler lab.

## Install

Install of the code here is a little heavy as we are using rather cutting edge packages, including cellpose, aicsimageio, and napari.

1) Create and activate a conda environment.

Assuming cellpose requires Python 3.8

```
conda create -y -n oligo-analysis python=3.8
```

Activate environment

```
conda activate oligo-analysis
```

Update pip

```
pip install --upgrade pip
```

2) Install from requirements.txt

```
requirements.txt
```

3) Some local installs are currently required. We will eventually remove this.

Make sure this is in my local dev branch

```
pip install ../napari-layer-table/.
```

## Running our pre-trained cellpose models on new image data

We have created some pre-trained models, they are in the `models/` folder.

To run the models on new image data and save a cellpose _seg.npy file, use `batchRunFolder()` in `oligoAnalysis/oligoAnalysisFolder.py`

This is required before you can **fully** use the main interface in `oligoanalysis/interface/oligoInterface.py`.


## Read datetime from czi

I have given up on this. When we switch to loading other files other than Zeiss czi, it won't work.

My original goal here was to load the date/time of image acquisition.

```
from aicsimageio import AICSImage

def readCzi():
    # looking for
    # Information|Document|CreationDate = 2022-09-22T15:03:58

    from aicspylibczi import CziFile

    cziPath = '/Users/cudmore/Dropbox/data/cudmore-fiji-plugins/test-data/shu-ling/pHA1_DsRed.czi'

    with open(cziPath) as f:
        czi = CziFile(f)

    #print('czi:', [x for x in dir(czi)])
    print('  czi.meta:')
    print(czi.meta)

    xpath_str = "./Metadata/Information/Document/CreationDate"
    _creationdate = czi.meta.findall(xpath_str)  #  [<Element 'CreationDate' at 0x7fa670647f40>]
    print('_creationdate:', _creationdate)
    for creationdate in _creationdate:
        # xml.etree.ElementTree.Element
        print('  creationdate:', creationdate)
        print(dir(creationdate))
        _datetime = creationdate.text
```

## Cellpose API

cellpose.io.add_model

## TODO

#### Nov 3, 2022

 - [done] Add histogram widget. Re-use histogram from PyMapManager
 
 - Add 'accept' column to napari-layer-table. This will be the start of editing from the table!
  - If point is selected and user hits keyboard 'a' then toggle 'accept' True/False
  - emit signal with point and row dict (including 'accept' column)
 
 - [done] swap in my table/model from napari-layer-table
  - hide lots of columns
  - make sure copy of table copies all columns

 - [done] make oligoAnalysis a package

 - make oligoanalysis a napari plugin

#### Nov 4, 2022

 - [done] make dark interface with qdarkstyle (see pymapmanager)
 - [done] reduce font size (see pymapmanager)
 - fix contrast slider signal/slot
    when oligointerface  gets slot contrast changed, check selected layer name and that it is an image layer and directly call layer.contrast_limits = [min, max]
 - when making oligoanalysis dataframe, pass entire df (all columns) and use hide() api to hide columns.
    goal is to get copy() to copy entire dataframe
 - in napari viewer, on edit 'accept', actually set oligo analysis label df
 - [done] on file selection, set gaussian sigma
 - implement 'set folder'
 - expand loaded file types. Right now we only do czi. Whatever we load, we need to know the number of channels

 - [done] in napari-layer-table, add api to set font size

 - [hard] if we have a cellpose model, run the model of merged rgb

 - rename repo to 'dapi-ring-analysis'. This is more general purpose than oligo

#### Nov 6 - back in Sac

 - Need to switch my naming of channels. Use 'DAPI' and 'CYTO' instead of 'green' and 'red'?

 
## As of Oct-31-2022, these are the installed python packages

```
Package                       Version
----------------------------- -----------
aicsimageio                   4.9.2
aicspylibczi                  3.0.5
alabaster                     0.7.12
anyio                         3.6.2
appdirs                       1.4.4
appnope                       0.1.3
argon2-cffi                   21.3.0
argon2-cffi-bindings          21.2.0
asciitree                     0.3.3
asttokens                     2.1.0
attrs                         22.1.0
Babel                         2.10.3
backcall                      0.2.0
beautifulsoup4                4.11.1
bleach                        5.0.1
build                         0.9.0
cachetools                    5.2.0
cachey                        0.2.1
cellpose                      2.1.0
certifi                       2022.9.24
cffi                          1.15.1
charset-normalizer            2.1.1
click                         8.1.3
cloudpickle                   2.2.0
commonmark                    0.9.1
dask                          2022.10.2
debugpy                       1.6.3
decorator                     5.1.1
defusedxml                    0.7.1
dnspython                     2.2.1
docstring-parser              0.15
docutils                      0.19
elementpath                   2.5.3
email-validator               1.3.0
entrypoints                   0.4
executing                     1.2.0
fasteners                     0.18
fastjsonschema                2.16.2
fastremap                     1.13.3
freetype-py                   2.3.0
fsspec                        2022.10.0
google-api-core               2.10.2
google-auth                   2.14.0
google-cloud-core             2.3.2
google-cloud-storage          2.5.0
google-crc32c                 1.5.0
google-resumable-media        2.4.0
googleapis-common-protos      1.56.4
HeapDict                      1.0.1
hsluv                         5.0.3
idna                          3.4
imagecodecs                   2022.9.26
imageio                       2.22.3
imagesize                     1.4.1
importlib-metadata            5.0.0
importlib-resources           5.10.0
ipykernel                     6.17.0
ipython                       8.6.0
ipython-genutils              0.2.0
ipywidgets                    8.0.2
jedi                          0.18.1
Jinja2                        3.1.2
jsonschema                    4.16.0
jupyter                       1.0.0
jupyter_client                7.4.4
jupyter-console               6.4.4
jupyter_core                  4.11.2
jupyter-server                1.21.0
jupyterlab-pygments           0.2.2
jupyterlab-widgets            3.0.3
kiwisolver                    1.4.4
llvmlite                      0.39.1
locket                        1.0.0
lxml                          4.9.1
magicgui                      0.6.0
MarkupSafe                    2.1.1
matplotlib-inline             0.1.6
mistune                       2.0.4
napari                        0.4.16
napari-console                0.0.6
napari-layer-table            0.0.10
napari-plugin-engine          0.2.0
napari-svg                    0.1.6
natsort                       8.2.0
nbclassic                     0.4.7
nbclient                      0.7.0
nbconvert                     7.2.3
nbformat                      5.7.0
nest-asyncio                  1.5.6
networkx                      2.8.7
notebook                      6.5.2
notebook_shim                 0.2.0
npe2                          0.6.1
numba                         0.56.3
numcodecs                     0.10.2
numpy                         1.23.4
numpydoc                      1.5.0
ome-types                     0.3.1
opencv-python-headless        4.6.0.66
packaging                     21.3
pandas                        1.5.1
pandocfilters                 1.5.0
parso                         0.8.3
partd                         1.3.0
pep517                        0.13.0
pexpect                       4.8.0
pickleshare                   0.7.5
Pillow                        9.3.0
Pint                          0.20.1
pip                           22.3
pkgutil_resolve_name          1.3.10
plotly                        5.11.0
prometheus-client             0.15.0
prompt-toolkit                3.0.31
protobuf                      4.21.9
psutil                        5.9.3
psygnal                       0.6.0
ptyprocess                    0.7.0
pure-eval                     0.2.2
pyasn1                        0.4.8
pyasn1-modules                0.2.8
pycparser                     2.21
pydantic                      1.10.2
Pygments                      2.13.0
PyOpenGL                      3.1.6
pyparsing                     3.0.9
PyQt5                         5.15.7
PyQt5-Qt5                     5.15.2
PyQt5-sip                     12.11.0
pyqtgraph                     0.13.1
pyrsistent                    0.19.1
python-dateutil               2.8.2
pytomlpp                      1.0.11
pytz                          2022.5
PyWavelets                    1.4.1
PyYAML                        6.0
pyzmq                         24.0.1
qtconsole                     5.3.2
QtPy                          2.2.1
requests                      2.28.1
resource-backed-dask-array    0.1.0
rich                          12.6.0
rsa                           4.9
scikit-image                  0.19.3
scipy                         1.9.3
Send2Trash                    1.8.0
setuptools                    65.5.0
six                           1.16.0
sniffio                       1.3.0
snowballstemmer               2.2.0
soupsieve                     2.3.2.post1
Sphinx                        5.3.0
sphinxcontrib-applehelp       1.0.2
sphinxcontrib-devhelp         1.0.2
sphinxcontrib-htmlhelp        2.0.0
sphinxcontrib-jsmath          1.0.1
sphinxcontrib-qthelp          1.0.3
sphinxcontrib-serializinghtml 1.1.5
stack-data                    0.6.0
superqt                       0.3.8
tenacity                      8.1.0
terminado                     0.17.0
tifffile                      2022.10.10
tinycss2                      1.2.1
tomli                         2.0.1
toolz                         0.12.0
torch                         1.13.0
tornado                       6.2
tqdm                          4.64.1
traitlets                     5.5.0
typer                         0.6.1
typing_extensions             4.4.0
urllib3                       1.26.12
vispy                         0.10.0
wcwidth                       0.2.5
webencodings                  0.5.1
websocket-client              1.4.1
wheel                         0.37.1
widgetsnbextension            4.0.3
wrapt                         1.14.1
xarray                        2022.10.0
xmlschema                     1.11.3
zarr                          2.13.3
zipp                          3.10.0
```