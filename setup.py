from setuptools import setup, find_packages

exec (open('oligoanalysis/version.py').read())

setup(
    name='oligoanalysis',
    version=__version__,
    description='',
    url='http://github.com/pymapmanager/OligoAnalysis',
    author='Robert H Cudmore',
    author_email='robert.cudmore@gmail.com',
    license='GNU GPLv3',
    #packages = find_packages(),
    packages=find_packages(include=['oligoanalysis',
                            'oligoanalysis.interface']),
    #packages = find_packages(exclude=['version']),
    #packages=[
    #    'pymapmanager',
    #    'pymapmanager.mmio'
    #],
    install_requires=[
            "numpy",
            "pandas",
            "tiffile",
            "scikit-image",
            "scipy",
            "aicsimageio",
            # aicsimageio czi reading requires extra install
            "aicspylibczi>=3.0.5",
            "fsspec>=2022.7.1",
            # dev
            "ipython",
            "jupyter",
            # interface
            "qdarkstyle",
            "pyqt5",
            "cellpose[gui]",
            "napari",
            "pyqtgraph",
            # for now, install locally with pip (in a dev branch)
            #napari_layer_table
            "plotly",
    ],
)

