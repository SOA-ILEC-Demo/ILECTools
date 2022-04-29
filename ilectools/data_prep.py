'''
Preprocessing tools for the 2018 and the 2021/12 data releases

Example: to download, decompress (unzip), and preprocess:

    data = prepare_all(URLS[2018], data_dir)

... for the set published in 2018.

parquet files are saved, which are loaded much faster than csvs.

'''


'''

Things I want to do

* Pass dataframe itself from one step to the next
* pass a directory and date and get that version
* pass a year and get that version

2017 file: 2001-13
2018 file: 2009-16
2021 file: 2009-18 but only ny for 2018

'''

# URLS gives the public URL by year in which the dataset was made public.
# I'm not providing context here, that context is on the appropriate site.
import pandas as pd, numpy as np, os, datetime as dt
import urllib
import urllib.request
from pathlib import Path
from typing import Union

import logging
from zipfile import ZipFile


# logging.basicConfig seems to be required, logging so unintuitive :/
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:  %(message)s', level=logging.ERROR)
# logger for just this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
URLS = {2021: 'https://cdn-files.soa.org/research/ilec/ilec-2009-18-20210528.zip',
        2018: 'https://cdn-files.soa.org/web/ilec-2016/ilec-data-set.zip',
        2017: 'http://cdn-files.soa.org/web/2009-13-indiv-life-ins-mort-exp-pivot.zip'
        }


def download(url: str, data_dir: Union[str, Path], force: bool=True) -> Path:
    """
    Download the dataset from the given url.

    Return the path of the extracted csv.

    Both files will be in the given directory.
    Time:
        might take 5min to downlad, 2min to unzip.
    """
    data_dir = Path(data_dir)
    fn = Path(url).name
    thezip = data_dir / fn # where the zip will land

    logger.info(f'Downloading {fn}')
    urllib.request.urlretrieve(url, thezip)
    logger.info('Download complete')
   
    logger.info(f'Decompressing {fn}')
    zf = ZipFile(thezip)
    if len(zf.filelist) >1:
        logger.warning('The archive contains multiple files: {}'.format('; '.join(zf.filelist)))

    i = 0
    fn_csv = Path(zf.filelist[i].filename)
    while (i < len(zf.filelist)) and fn_csv.suffix.lower() != '.csv': # keep looking
        i+=1
        fn_csv = Path(zf.filelist[i].filename)

    # Make sure found a csv
    if fn_csv.suffix.lower() != '.csv':
        logger.error('No CSVs in the archive, ending')
        return
    zf.extract(member=str(fn_csv), path=str(data_dir)) # extract into that same directory
    logger.info(f'Decompressed {fn_csv}')

    return data_dir / fn_csv

def read_csv(fn_csv: Union[str, Path], chunksize: int=500000) -> pd.DataFrame:
    """Read in  the downloaded csv at the given filename and return a dataframe.
    This function specifies data types of certain ambiguous columns and reads in chunks.
    It also saves a pkl and a parquet of the data, with no changes in the column names, just as read in.
    """
    fn_csv = Path(fn_csv) # make sure have Path
    logger.info(f'Reading original csv {fn_csv}')
    # the number of row in each data frame read in as a chunk
    i = 0
    # the list that contains all the dataframes
    list_of_dataframes = []
    for df in pd.read_csv(fn_csv, chunksize=chunksize,
        # a few columns have nulls and need a specified type (float64), others do not need the full default integral type.
        dtype={ fld:dtyp
                for dtyp, flds
                in [(np.float64, 'Cen3MomP1wMI_byAmt,Cen3MomP2wMI_byAmt,Cen3MomP1_byAmt,Cen3MomP2_byAmt,Cen3MomP3wMI_byAmt,Cen3MomP3_byAmt'),
                    (np.float64, 'Number_Of_Preferred_Classes,Preferred_Class'), # these contain nulls in the original so make float
                    (np.int8,    'Issue_Age,Duration,Attained_Age,Preferred_Indicator'),
                    (np.int16, 'Issue_Year,Observation_Year')]
                for fld in flds.split(',')
                }
                 ):
        logger.info(f'{i*chunksize:,.0f} records')
        i+=1
        if df.columns[0].lower() != 'observation_year': # is a row number, make it index
            df = df.set_index(df.columns[0])
        list_of_dataframes.append(df)
    logger.info('... read.  Concatenating:')
    data = pd.concat(list_of_dataframes)
    logger.info(f"...concatenated, {len(data):,.0f} records in result.")
    data.index.name = 'row' # as was unnamed in original file
    save_pkl_parquet(data, fn_csv)
    return data


def save_pkl_parquet(data: pd.DataFrame, fn_csv: Union[str, Path]):
    """Save the data frame as a pickle and as a parquet, both of which are much faster
    to read and include data type info.

    Pass the filename from which the csv was read.

    The pickle will have the extension pkl, and the parquet will have parquet, saved in the same directory.
    """
    fn_csv = Path(fn_csv) # make sure get a Path object
    # save as a pickle
    logger.info('Saving pkl...')
    data.to_pickle(fn_csv.with_suffix('.pkl'), protocol=4)
    logger.info('...saved.')

    # save nonpartitioned parquet
    logger.info('Saving parquet...')
    data.to_parquet(fn_csv.with_suffix('.parquet'), engine='pyarrow')
    logger.info('... saved.')


def make_v1(data: pd.DataFrame, fn_csv: Union[str, Path])->pd.DataFrame:
    """Make v1 of the data, save parquet, and return it.
    1. lower-case field (column) names with underscores instead of spaces: renamed as with utilities.NEW_NAMES
    2. Fill preferred_class and number_of_preferred_classes with 1s instead of nulls, convert to smaller type
    3. Makes field 'uw' that is smoker status / number of preferred classes / preferred class
    4. Make columns band_min, band_max for max and min of face band
    5. Strip leading and trailing spaces from products
    6. Make smoker_status proper case so capitalization is consistent with Unknown and UNKNOWN and no caps inSide a wORd

    Saves as
        _v1.parquet
        _v1.pkl using pickle  protocol 4
    """
    data.columns = [c.lower().replace(' ', '_') for c in data.columns] # is faster than rename
    from .utilities import NEW_NAMES
    data.columns = [NEW_NAMES.get(c,c) for c in data.columns] # renaming in place was slow
    data['insurance_plan'] = data['insurance_plan'].str.strip()
    data['smoker_status'] = data['smoker_status'].str.title()
    # convert preferred fields to integers, must plug nulls, 1 makes sense
    for c in ['preferred_class', 'number_of_preferred_classes']:
        data[c] = data[c].fillna(1).astype(np.int8)

    logger.info('set uw column')
    data['uw'] = (data['smoker_status'].str[0] 
                      +'/'+ data['number_of_preferred_classes'].astype(str)
                      +'/'+ data['preferred_class'].astype(str))
    logger.info('set band min, band max')
    fb = set(data.face_amount_band) # the distinct values

    # Dictionaries for low and high amounts: 
    fba = {_b:float(_b.replace('+','-1000000000').split('-')[0].strip()) for _b in fb}
    fbb = {_b:float(_b.replace('+','-1000000000').split('-')[1].strip()) for _b in fb}
    data['band_min'] = data['face_amount_band'].map(fba)
    data['band_max'] = data['face_amount_band'].map(fbb)

    logger.info('Saving parquet...')
    fnv1 = (fn_csv.parent / (fn_csv.stem + '_v1')).with_suffix('.parquet')
    data.to_parquet(fnv1, engine='pyarrow')
    logger.info(f'Saved v1: {fnv1}')
    return data


def make_v2(data: pd.DataFrame, fn_csv: Union[str, Path]) -> pd.DataFrame:
    """Make v2 of data and save as parquet.

    data must be output from make_v1.

    1. Includes only adults
    2. Excludes columns for moments and for industry tables besides 2015vbt
    """

    logger.info('Dropping rows for minors')
    data = data[(data.issue_age>17)]
    logger.info('Dropping columns for moments, industry tables besides 2015vbt')
    import re
    data = data.drop(columns=[c for c in data.columns if re.match(r'.*(7580|2001|2008|_cen\d)', c)])
    logger.info('Saving as parquet')
    fnv2 = (fn_csv.parent / (fn_csv.stem + '_v2')).with_suffix('.parquet')
    data.to_parquet(fnv2, engine='pyarrow')
    logger.info(f'Saved v2: {fnv2}')
    return data

def prepare_all(url: Union[str, Path], data_dir: Union[str, Path])->pd.DataFrame:
    """Download the zip from the url and prepare the parquet files."""
    assert(Path(url).suffix.lower()=='.zip')
    fn_csv = download(url, data_dir)
    d = read_csv(fn_csv)
    d = make_v1(d, fn_csv)
    d = make_v2(d, fn_csv)
    return d

