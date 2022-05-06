import IPython.display as dis

import re, numpy as np, pandas as pd, matplotlib.pyplot as plt, itertools, operator
import matplotlib as mpl, matplotlib.pyplot as plt
from datetime import datetime as dt
#http://statsmodels.sourceforge.net/devel/gettingstarted.html
import statsmodels.api as sm, statsmodels.formula.api as smf, patsy
from patsy import dmatrices
import networkx as nx

from functools import reduce


def add_cols(df):
    '''Add certain columns to the dataframe for modeling and return it.
    These additional columns are categorical groupings of fields:
    ltp: level term period grouping 20 year and not term, to 20 year will be a baseline term rate
    dur_band1
    ia_band1
    iy_band1
    '''

    # I don't like 'Not Level Term' and "N/A (Not Term)" both being in there, should probably have consolidated them
    # level term period for this purpose
    df['ltp'] = (df.soa_anticipated_level_term_period.replace(
        {'20 yr anticipated': '20 yr or N/A (Not Term)',
         'N/A (Not Term)': '20 yr or N/A (Not Term)'}
    ).str.replace(' anticipated', '')
                 )

    # duration bands in the original paper
    if not 'dur_band1' in df and 'duration' in df:
        df['dur_band1'] = 'n/a'  # a default, which will get overridden
        for band, durs in {
            '01': [1],
            '02': [2],
            '03': [3],
            '04-05': [4, 5],
            '06-10': range(6, 11),
            '11-15': range(11, 16),
            '16-20': range(16, 21),
            '21-25': range(21, 26)}.items():
            df.loc[df.duration.isin(durs), 'dur_band1'] = band
        df.loc[df.duration > 25, 'dur_band1'] = '26-150'

    if not 'ia_band1' in df and 'issue_age' in df:
        df['ia_band1'] = '18-24'
        for a in range(25, 95, 5):
            df.loc[df.issue_age.between(a, a + 4), 'ia_band1'] = f'{a}-{a + 4}'
        df.loc[df.issue_age > 94, 'ia_band1'] = '95+'
        if df.issue_age.min() < 18:
            df.loc[df.issue_age < 18, 'ia_band1'] = '00-18'

    if not 'iy_band1' in df and 'issue_year' in df:
        df['iy_band1'] = '1900-1989'
        df.loc[df.issue_year.between(1990, 1999), 'iy_band1'] = '1990-1999'
        df.loc[df.issue_year.between(2000, 2009), 'iy_band1'] = '2000-2009'
        df.loc[df.issue_year >= 2010, 'iy_band1'] = '2010-'

    return df


def exhibit_g_subset(data, max_observation_year=2013):
    '''
    Return the subset used for Exhibit G:
    * post level term indicator not 'Post Level term'
    * select period i.e. durations 1-25
    * adult issue ages, > 17
    * observation year <= max_observation_year, 2013 by default for consistency with report

    Also return only those records with exposure.  There is a trivial amount claim, no count, in a record
    with no exposure.
    '''

    # subset of data for exhibit g, the example used in
    return data[(data.soa_post_level_term_indicator != 'Post Level Term')
              & (data.duration < 26)  # select only
              & (data.observation_year <= max_observation_year)  # for consistency with the report, update if you like
              & (data.issue_age > 17)  # no juvies pls
              & (data.amount_exposure > 0) # why there anyway... and would crash lm factor regression
              & (data.policy_exposure > 0) # ditto
            ]

def getKV(k):
    """Get key, value from label, value in the parameters"""
    if k=='Intercept':
        return ('Intercept','Intercept')
    m = re.match(r"^C\((.*)\)\[(.*.)]$", k)  # has C in name to show is a category
    if m: #has C()
        m  = m.groups()
        m1 = list(m[1:])
        res = tuple([m[0].split(',')[0]]+m1)  # remove "treatment" if there in 1st one
    else:
        m = re.match(r"^(.*)\[(.*.)]$", k)  # lacks C showing cateogry

        if m:
            res = m.groups()
        else:
            return k # just don't do anything
            # If field value is an integer then return it.
    if res[1][:2]=='T.':
        res = (res[0], res[1][2:]) # Strip the T. at the start of the value, I'm not sure why it's there.
    if re.match('^[\d]*$', res[1]) and res[1][0]!='0': # keep leading zeros if any, works better w/keys I have so far
        res = (res[0], int(res[1]))
    return res

def getKVs(k):
    """Get the KVs where multiple cross combinations are present."""
    if ':' in k: 
        return pd.Series({_k: _v for _k, _v in map(getKV, k.split(':'))})
    else:
        return getKV(k)


class PoissonWrapper(object):
    """
    This class adds a few presentation functions to the statsmodels tools.

    Attributes:
        offset_column: the name of the column being adjusted
        univariate_factors: whether the formula contains only univariate factors, i.e. no cross combinations.
            The presentation tools to not allow cross combinations as they were not needed for the 2018 report.
        fit: the results of statsmodels.formula.api.poisson fit
        ... and its properties below it
        fit.model
        fit.model.data.frame  The source dataframe, so not necessary to keep it separately, it's saved here
        fit.model.formula     The formula
            ... etc ... all the attributes of the fit, as that result is just svaed here

    functions:

    """

    def __init__(self, formula, data, offset_column):
        """pass:ass either:
            fit: a fit object from a statsmodels glm
        or:
            data, formula, offsetcol

        formula: the patsy formula for the model
        data: the dataframe
        offset_column: the column adjusted multiplicatively by factor for fit
        """
        # You could just use smf.glm here but also specify family parameter

        self.univariate_factors = not (':' in formula or '*' in formula)
        try:
            assert self.univariate_factors
        except AssertionError:
            print('No cross combinations for these presentation tools please, your formula contains * or :.'
                  '  The model will be run but the presentation tools will not work.\n')

        self.fit = smf.poisson(data=data,
                      formula=formula,
                      offset=np.log(data[offset_column]),
                      #family=sm.families.Poisson(link=sm.families.links.log())
                      ).fit()

        self.offset_column = offset_column  # to keep the name


    def pretty_params(self):
        """Prettify the parameters: also add 1 where don't have one."""
        vals = {getKV(_k):_v
                for _k,_v
                in self.fit.params.to_dict().items()
                }
        s = pd.Series(vals)
        data = self.fit.model.data.frame # for convenience
        # For each column, get any missing values - such as defaults, set their factor to 1
        # (not for Intercept - there's no corresponding column)
        for c in s.index.levels[0]:
            if c!='Intercept':
                vals.update({(c, v):0. # append to dictionary a factor of 0 for each default category, i.e. not lised
                             for v
                             in list(set(data[c]) # all values of category c
                                     .difference(set(s[c].index)) # ... removing those for which have factor
                                     )
                             })
        return pd.Series(vals).sort_index()


    def compare_factors_with_a_to_t(self):
        """Compare factors and A/Model in aggregate for the set.
        NOTE: the factors are all of the model factors that interrelate.  For example, if there
            are factors for combinations of face band and underwriting class, but also for
            combinations of underwriting class and sex, then all three are compared together.
        """
        # column names for actual and tabluar

        if not self.univariate_factors:
            return

        # The coefficients
        coef = np.exp(self.pretty_params())
        coef = coef.loc[~coef.index.isin([('Intercept', 'Intercept')])]


        a, t = self.fit.model.formula.split('~')[0].strip(), self.offset_column # actual, tabular columns in data
        data = self.fit.model.data.frame  # The data, for convenience

        # variables for which have categorical factors and will compare to a/tab ratio, makes sure no Intercept
        vars = list(set(coef.index.to_frame()[0]))
        a_t = pd.concat([data.pivot_table(index=i, values=[a,t], aggfunc=np.sum)
                         for i in vars],
                        keys=vars)
        a_t = a_t[a] / a_t[t] # the ratio is all we want here

        return pd.DataFrame({'Factor':coef, 'A/Table':a_t}).sort_index()

    def plot_comparision(self):
        '''Return dictionary of plots of a/table and factor'''

        plots = {}
        plotoptions = dict(rot=45, ylim=(0.5, 2), grid=True)
        a_t = self.compare_factors_with_a_to_t()
        vars = list(set(a_t.index.to_frame()[0])) # get variables to show

        for i in vars:
            ax = a_t.loc[i].plot(style=['r.-', 'b.-'], **plotoptions)
            plt.axhline(1, color='k')
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
            plots[i] = ax.figure
            plt.close()
        return plots


    def comp_factor_dist(self, index_column, for_each, data=None, plot=False, **kwargs):
        """
        Return:
            distribution of offset column (exposure, tabular expected, etc)  across index column
            values for each separate value of field "for_each"
        args:
            data=None: default is model's data if none specified
            of_column=None: column of which to take the distribution, offset column used if none specified

        Example: where t is an instance:  plot ia_band1 factors on secondary y axis, 
                    and distribution of each insurance_plan by ia_band1
            t.comp_factor_dist('ia_band1', 'insurance_plan').plot(secondary_y='Factor')
        """
        # comments from retired subsidiary function
        """
                Align factor with distribution between categories specified by indexColumn,
                 for item in ofColumn (such as expected claims or exposure),
                 splitting distribution in for_each, 
                 really only works for log link.

                 Returns: dataframe, split by indexColumn down rows (dataframe index); 
                     Columns: 'Factor' for factor of indexColumn, 
                             then a column for each elelement of column forEach (such as for each insurance_plan)
                 """

        # distribution: columns add to 100%
        dist = self.fit.model.data.frame.pivot_table(index=index_column, columns=for_each,
                                                     values=self.offset_column,
                                                     aggfunc=np.sum).apply(lambda s: s/s.sum())
        # Get the factor for indexColumn
        fact = pd.DataFrame({'Factor': np.exp(self.pretty_params().loc[index_column])})

        if plot:
            fig, ax = plt.subplots(1)
            dist.plot(kind='bar', ax=ax, rot=90, title='Distribution of {}'.format(ofColumn))
            fact.plot(secondary_y='Factor', ax=ax, rot=90)
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
        return pd.concat([fact, dist], axis=1).fillna(0)

    def comp_factor_avg(self, for_each, data=None, of_column=None):
        """Compare average factors weighted by the fit's offset column for different splits in for_each.
        arguments:
            for_each: category to be split into separate items
            data=None: data to split, use model's dataset if none provided
            of_column=None: column by which to weight the factors to get the average,
                        such as exposure or expected claims;
                        default is model's offset column if none is specified.
            """
        if data is None:
            data =  self.fit.model.data.frame
        if of_column is None:
           of_column = self.offset_column

        # To avoid repeated calls
        prepar = self.pretty_params()

        # List the levels in the model to compare. All of them but the intercept and the one for which we'll show
        # the factor, since for that one the average is just the factor.
        
        lvls = list(set(prepar.index.levels[0]).difference(['Intercept', for_each]))
        
        #function to get weighted average of factors.
        # The dataframe passed hasthe factor as the 1st column, and is output of comp_factor_dist.
        avgFactor = lambda df: df[[c for c in df.columns if c != 'Factor']].multiply(df['Factor'], axis=0).sum() 
        res = pd.DataFrame({lvl:avgFactor(self.comp_factor_dist(lvl, for_each, plot=False,
                                                                data=data, ofColumn=of_column))
                            for lvl in lvls})  # get the average factor for each lvl
        return pd.concat([ # Get the factors themselves
                          #self.fit.family.link.inverse(pd.DataFrame({forEach:prepar[forEach][list(set(data[forEach]))]}))
                           np.exp(pd.DataFrame({for_each:prepar[for_each]}))
                            #add column for factors for thing we're splitting by
                         , res],
            axis=1).T #append factors for thing we're splitting by and return.

    def factor_analysis_exhibit(self, for_each):
        """Show across categories in for_each:
        
        A/Table
        Factor
        Average of all other factors
        Approximate A/Table (product of all factors or average thereof) 
       
       arguments:
           forEach: should be a category in the model
           
       ... Compare factors for a split with the average factor in the applicable other categories and a/table"""
        cf = self.compare_factors_with_a_to_t().loc[for_each].T

        # the average factor: must exclude category forEach from this becasue is shown as factor already
        cfa = self.comp_factor_avg(for_each)
        # Show intercept (overall factor) and take out the one that we're splitting by since we show it already
        cfa = pd.concat([
            pd.DataFrame(np.exp(self.pretty_params()[('Intercept', 'Intercept')])
                             , index=['Overall']
                         , columns=cfa.columns)
            , cfa.loc[cfa.index!= for_each]], axis=0
            )
        # datafame of a/t and the estimate
        atdf = pd.DataFrame({'Observed':cf.loc['A/Table']
                             , 'Approximated':cfa.product(axis=0) * cf.loc['Factor']})[
                            ['Observed','Approximated'] # to reorder
                        ].T
        res = pd.concat([atdf # The A/T that we'd like to explain
                         , pd.DataFrame({for_each:cf.loc['Factor']}).T # the factors themselves
                         , cfa
                         ]
                         , keys=['A/VBT15', 'Factor','Avg factors']) # The overall factor (intercept) will show as avg, is OK
        res.index.names = ['',''] # so don't show "None","None" when writing out
        return res


def get_exhibit_g(data, metric, basis='2015vbt'):
    splits=  ['face_amount_band', 'dur_band1']
    res = []
    for s in splits:
        pt = data.pivot_table(index=s,
                                columns='insurance_plan',
                                values=[f'{metric}_actual', f'{metric}_{basis}'],
                                aggfunc=np.sum, margins=True)
        res.append(pt[f'{metric}_actual'] / pt[f'{metric}_{basis}'])
    return pd.concat(res, keys=splits).style.format('{:,.0%}')
