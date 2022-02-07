'''
Preprocessing tools for the 2021/12 data release

Original data is at https://cdn-files.soa.org/research/ilec/ilec-2009-18-20210528.zip



'''

import pandas as pd, numpy as np, os, datetime as dt



import urllib
import urllib.request
from pathlib import Path
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)


from zipfile import ZipFile


def download(data_dir):
    """
    Download the dataset into the given directory, might take 5min to downlad, 2min to unzip
    """
    data_dir =  Path(data_dir)
    fn  = "ilec-2009-18-20210528.zip"
    thezip = data_dir / fn
    logging.info(f'Downloading {fn}')
    urllib.request.urlretrieve(f"https://cdn-files.soa.org/research/ilec/{fn}", thezip)
    logging.info('Download complete')
    logging.info('Decompressing')
    zf = ZipFile(thezip)
    zf.extractall(data_dir)
    logging.info('Decompressed')
    return data_dir / zf.filelist[0].filename # is only one file

