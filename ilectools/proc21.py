'''
Preprocessing tools for the 2021/12 data release

Original data is at https://cdn-files.soa.org/research/ilec/ilec-2009-18-20210528.zip

Package constant fn_csv is the file in the zip.

'''

import pandas as pd, numpy as np, os, datetime as dt



import urllib
import urllib.request
from pathlib import Path
from typing import Union

import logging
# logging.basicConfig seems to be required, logging so unintuitive :/
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:  %(message)s', level=logging.ERROR)
# logger for just this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from zipfile import ZipFile
fn_csv = 'ILEC 2009-18 20210528.csv' # the fn of the csv

def test(msg):
    logger.info(msg)

def download(data_dir: Union[str, Path], force: bool=True) -> Path:
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
    thezip = download_dir / fn

    logger.info(f'Downloading {fn}')
    urllib.request.urlretrieve(f"https://cdn-files.soa.org/research/ilec/{fn}", thezip)
    logger.info('Download complete')
   
    # Unzip
    logger.info(f'Decompressing {fn}')
    zf = ZipFile(thezip)
    assert(zf.filelist[0].filename==fn_csv) # this is the file that should be in the zip
    zf.extractall(download_dir) # extract into that same directory
    logger.info('Decompressed')
    return download_dir / fn_csv # is only one file, 'ILEC 2009-18 20210528.csv'


def read_csv(data_dir: Union[str, Path], chunksize: int=500000) -> pd.DataFrame:
    """Read in  the downloaded csv at the given filename  and return a dataframe.
    This function specifies data types of certain ambigous columns and reads in chunks.
    """
    data_dir = Path(data_dir)
    ffn_csv = data_dir / fn_csv
    logger.info('Reading original csv')
    # the number of row in each data frame read in as a chunk
    i = 0
    # the list that contains all the dataframes
    list_of_dataframes = []
    for df in pd.read_csv(ffn_csv, chunksize=chunksize, 
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
        list_of_dataframes.append(df.set_index(df.columns[0])) # the 1st column is a row number, which we ought to keep for reference
    logger.info('... read.  Concatenating:')
    data = pd.concat(list_of_dataframes)
    logger.info(f"...concatenated, {len(data):,.0f} records in result.")
    data.index.name = 'row' # as was unnamed in original file
    return data


def save_pkl_parquet(data: pd.DataFrame, data_dir: Union[str, Path]):
    """Save the data frame as a pickle and as a parquet, both of which are much faster to read and include data type info.
    Pass the filename from which the csv was read.
    The pickle will have the extension pkl, and the parquet will have parquet.
    """
    data_dir = Path(data_dir)
    ffn_csv = data_dir / fn_csv
    # save as a pickle
    logger.info('Saving pkl...')
    data.to_pickle(ffn_csv.with_suffix('.pkl'), protocol=4)
    logger.info('...saved.')

    # save nonpartitioned
    logger.info('Saving parquet...')
    data.to_parquet(ffn_csv.with_suffix('.parquet'), engine='pyarrow')
    logger.info('... saved.')


def make_v1(data: pd.DataFrame, data_dir: Union[str, Path])->pd.DataFrame:
    """Make v1 of the data, save parquet, and return it.
    1. lower-case names with underscores instead of spaces
    2. Fill preferred_class and number_of_preferred_classes with 1s instead of nulls, convert to smaller type
    3. Makes field 'uw' that is smoker status / number of preferred classes / preferred class
    4. Make columns band_min, band_max for max and min of face band
    Saves as parquet file _v1.parquet
    """
    data.columns = [c.lower().replace(' ', '_') for c in data.columns]
    # convert preferred fields to integers, must plug nulls, 1 makes sense
    for c in ['preferred_class', 'number_of_preferred_classes']:
        data[c] = data[c].fillna(1).astype(np.int8)

         
    # Renaming columns for convenience and consistency of name components (metric_item where metric = policy or amount)
    newnames = {'number_of_deaths':'pol_act'
                 , 'death_claim_amount':'amt_act'
                 , 'policies_exposed':'pol_xps'
                 , 'amount_exposed':'amt_xps'
                 , 'expdeathqx2015vbtwmi_bypol':'pol_2015vbtwmi'
                 , 'expdeathqx2015vbtwmi_byamt':'amt_2015vbtwmi'}
    newnames.update({_:'{}_{}'.format({'amount':'amt', 'policy':'pol'}[_.split('_')[-1]],
                                           _.split('_')[2][2:])
                     for _ in ['expected_death_qx7580e_by_amount',
                                'expected_death_qx2001vbt_by_amount',
                                'expected_death_qx2008vbt_by_amount',
                                'expected_death_qx2008vbtlu_by_amount',
                                'expected_death_qx2015vbt_by_amount',
                                'expected_death_qx7580e_by_policy',
                                'expected_death_qx2001vbt_by_policy',
                                'expected_death_qx2008vbt_by_policy',
                                'expected_death_qx2008vbtlu_by_policy',
                                'expected_death_qx2015vbt_by_policy'
                                  ]})

    # central moments
    newnames.update({c:'{}_{}'.format(c[-3:], c.split('_')[0]) 
      for c in ['cen2momp1wmi_byamt',
                  'cen2momp2wmi_byamt',
                  'cen3momp1wmi_byamt',
                  'cen3momp2wmi_byamt',
                  'cen3momp3wmi_byamt',
                  'cen2momp2wmi_bypol',
                  'cen3momp3wmi_bypol',
                  'cen2momp1_byamt',
                  'cen2momp2_byamt',
                  'cen3momp1_byamt',
                  'cen3momp2_byamt',
                  'cen3momp3_byamt',
                  'cen2momp2_bypol',
                  'cen3momp3_bypol']})

    data.columns = [newnames.get(c,c) for c in data.columns] # renaming in place was slow 

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

    logger.info('saving parquet...')
    data.to_parquet(Path.joinpath(Path(data_dir), 'ILEC 2009-18 20210528_v1.parquet'), engine='pyarrow')
    return data


def make_v2(data: pd.DataFrame, data_dir: Union[str, Path]) -> pd.DataFrame:
    """Make v2 of data and save as parquet.
    1. drop rows for minors, i.e. keep only adjusts, issue ages 18+
    2. drop rows for study years >2017, i.e. 2018
    2. drop columns for moments and industry tables besides 2015vbt
    """
    logger.info('Dropping rows for minors, observation year >2017')
    data = data[(data.issue_age>17) & (data.observation_year<=2017)]
    import re
    logger.info('Dropping columns for moments, industry tables besides 2015vbt')
    data = data.drop(columns=[c for c in data.columns if re.match(r'.*(7580|2001|2008|_cen\d)', c)])
    logger.info('Saving as parquet')
    data.to_parquet(Path.joinpath(Path(data_dir), 'ILEC 2009-18 20210528_v2.parquet'), engine='pyarrow')
    logger.info('v2 saved.')
    return data


def prepare_all(data_dir: Union[str, Path])->pd.DataFrame:
    """Download the file and prepare the parquet files."""
    
    download(data_dir)
    d = read_csv(data_dir)
    d = make_v1(d, data_dir)
    d = make_v2(d, data_dir)
    return d
