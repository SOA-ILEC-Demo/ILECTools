'''
Preprocessing tools for the 2021/12 data release

Original data is at https://cdn-files.soa.org/research/ilec/ilec-2009-18-20210528.zip



'''

import pandas as pd, numpy as np, os, datetime as dt



import urllib
import urllib.request
from pathlib import Path
import logging
from typing import Union

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)


from zipfile import ZipFile


def download(data_dir: str, force: bool=True) -> Path:
    """
    Download the dataset from 
    
        https://cdn-files.soa.org/research/ilec/ilec-2009-18-20210528.zip
    
    and decompress 'ILEC 2009-18 20210528.csv' contained therein.

    Both files will be saved into the directory.

    Time:
        might take 5min to downlad, 2min to unzip.

    Return the full path of the unzipped csv file as Path object
    """

    # download
    download_dir =  Path(data_dir) # to turn into Path instance
    fn  = "ilec-2009-18-20210528.zip"
    fn_csv = "ILEC 2009-18 20210528.csv"

    thezip = download_dir / fn
    logging.info(f'Downloading {fn}')
    urllib.request.urlretrieve(f"https://cdn-files.soa.org/research/ilec/{fn}", thezip)
    logging.info('Download complete')
   
    # Unzip
    logging.info(f'Decompressing {fn}')
    zf = ZipFile(thezip)
    assert(zf.filelist[0].filename==fn_csv) # this is the file that should be in the zip
    zf.extractall(download_dir) # extract into that same directory
    logging.info('Decompressed')
    return download_dir / fn_csv # is only one file, 'ILEC 2009-18 20210528.csv'





def read_csv(fn_csv: Union[str, Path], chunksize: int=500000) -> pd.DataFrame:
    """Read in  the downloaded csv at the given filename  and return a dataframe.

    This function specifies data types of certain ambigous columns and reads in chunks.
    """

    logging.info('Reading original csv')
    # the number of row in each data frame read in as a chunk
    i = 0
    # the list that contains all the dataframes
    list_of_dataframes = []
    for df in pd.read_csv(fn_csv, chunksize=chunksize, 
        # a few columns have nulls and need a specified type
        dtype={c:np.float64  
                for c 
                in 'Cen3MomP1wMI_byAmt,Cen3MomP2wMI_byAmt,Cen3MomP1_byAmt,Cen3MomP2_byAmt,Cen3MomP3wMI_byAmt,Cen3MomP3_byAmt'.split(',')}
                 ):
        logging.info(f'{i*chunksize:,.0f} records')
        i+=1
        list_of_dataframes.append(df.set_index(df.columns[0])) # the 1st column is a row number, which we ought to keep for reference
    logging.info('... read.  Concatenating:')
    dat = pd.concat(list_of_dataframes)
    logging.info(f"...concatenated, {len(dat):,.0f} records in result.")
    dat.index.name = 'row' # as was unnamed in original file
    return dat


def save_pkl_parquet(data: pd.DataFrame, fn_csv: Union[str, Path]):
    """Save the data frame as a pickle and as a parquet, both of which are much faster to read and include data type info.
    Pass the filename from which the csv was read.
    The pickle will have the extension pkl, and the parquet will have parquet.
    """

    # save as a pickle
    logging.info('Saving pkl...')
    data.to_pickle(fn_csv.with_suffix('.pkl'), protocol=4)
    logging.info('...saved.')

    
    # save nonpartitioned
    logging.info('Saving parquet...')
    data.to_parquet(fn_csv.with_suffix('.parquet'), engine='pyarrow')
    logging.info('... saved.')
