# ILEC Tools

Tools for processing and reporting on ILEC data, especially those used for 
* Data preparation: e.g. to strip leading spaces from product names, save as faster-loading binary file in [parquet format](https://en.wikipedia.org/wiki/Apache_Parquet)
  * Preprocessing dataset [published in 2021](https://github.com/SOA-ILEC-Demo/ILECTools/blob/master/sample_notebooks/preprocess_2021_dataset.ipynb)
  * Preprocessing dataset [published in 2017](https://github.com/SOA-ILEC-Demo/ILECTools/blob/master/sample_notebooks/preprocess_2017_dataset.ipynb)
* [2019 ValAct presentation](https://bdholland.com/pres/201908_ValAct/2019_ValAct_76_Holland_02.pptx) on analytics efforts
  * See sample notebook [2019_ValAct_demo.ipynb](https://github.com/SOA-ILEC-Demo/ILECTools/blob/master/sample_notebooks/2019_ValAct_demo.ipynb)
* [Beyond Actual to Table](https://www.soa.org/resources/research-reports/2018/beyond-actual-table/) paper 
  * Related: 2017 Annual Meeting [slides](https://www.soa.org/globalassets/assets/files/research/exp-study/2017-new-directions-in-experience-studies.pptx)
  * See sample notebook [Beyond_Actual_to_Table.ipynb](https://github.com/SOA-ILEC-Demo/ILECTools/blob/master/sample_notebooks/Beyond_Actual_to_Table.ipynb)
* *To do*: [2017 Individual Life Insurance Mortality Experience Report ](https://www.soa.org/research/topics/indiv-val-exp-study-list/)
    * duplicate figures... a notebook for this would be nice


Subpackages
* ```data_prep```: downloading, preprocessing 2021 data
* ```rollforward```: rollforward computation and presentation, used for the 2019 ValAct presentation
* ```decomp_trend```: Trend of study year exposure or other metrics by decomposition, used for the 2019 ValAct presentation
* ```a2t```

See the notebooks in the directory ```sample_notebooks``` for examples of use.

## Installation

Please use an environment manager.  To leave your environment untouched you can add the ILECTools 
directory to your system path:
```
import sys
sys.path.append('ILECTools') # specify the path as needed
import ilectools
# now you're good to go
```

