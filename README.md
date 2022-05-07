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

## Environment management

An "environment" is a set of various packages, all of which have their own versions.  Not all 
package versions are compatible with each other so it is important to keep track of what 
you are doing.   The notebooks have been tested in an environment which conda-users 
should be able to duplicate  with the included environment specification files in the
```env``` subdirectory.  I intend for  you to be able to create the ```ilec0``` environment 
on your machine with this command:

```commandline
conda env create -f environment.yml # of course include any path you need to the yml file
```

These sample notebooks were run on machines with 32gb RAM, Ubuntu 20.04, with a 32gb 
swapfile.  Your mileage may vary!  You can see in the environment specs that Python 3.10.4 
was used.  Miniforge was used for environment management, as opposed to miniconda or the
Anaconda python distribution.  Pay attention to your licenses always.

For more on environment management with ```conda```:
* https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
* https://shandou.medium.com/export-and-create-conda-environment-with-yml-5de619fe5a2


