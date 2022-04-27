# ILEC Tools

Tools for processing and reporting on ILEC data, especially those used for 
* Data preparation: e.g. to strip leading spaces from product names, save as faster-loading binary file
* [2019 ValAct presentation](https://bdholland.com/pres/201908_ValAct/2019_ValAct_76_Holland_02.pptx) on analytics efforts
* [Beyond Actual to Table](https://www.soa.org/resources/research-reports/2018/beyond-actual-table/) paper 
  * for which also see 2017 Annual Meeting [slides](https://www.soa.org/globalassets/assets/files/research/exp-study/2017-new-directions-in-experience-studies.pptx)
* *To do*: [2017 Individual Life Insurance Mortality Experience Report ](https://www.soa.org/research/topics/indiv-val-exp-study-list/)
    * duplicate figures


Subpackages
* ```data_prep```: downloading, preprocessing 2021 data
* ```rollforward```: rollforward computation and presentation
* ```decomp_trend```: Trend of study year exposure or other metrics by decomposition
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

