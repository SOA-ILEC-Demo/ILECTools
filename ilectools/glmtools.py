import IPython.display as dis

import re, numpy as np, pandas as pd, matplotlib.pyplot as plt, itertools, operator
import matplotlib as mpl
from datetime import datetime as dt
#http://statsmodels.sourceforge.net/devel/gettingstarted.html
import statsmodels.api as sm, statsmodels.formula.api as smf, patsy
from patsy import dmatrices
pd.options.display.max_rows=255

def ff(fmt):
    pd.options.display.float_format = fmt.format

# http://stackoverflow.com/questions/19470099/view-pdf-image-in-an-ipython-notebook
class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size

  def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

#I need the correct face band ordering.
faceBands = [
    ' <100000',
    ' 100000-249999',
    ' 250000-499999',
    ' 500000-999999',
    '1000000-2499999',
    '2500000+']


def getSummary(data, indexColumns, *args, **kwargs):
    """Make a pivot table of data by columns using the known value columns by policy as values;
    resetting index and removing bogus zero or null values.

    Working with the nonaggregated data is too slow, this function aggregates the dataframe with a pivot
    and then resets the index.
    It only keeps data by count.

    Arguments:

data    : the data to list: offset must be valid values, e.g. nononzero for log link
indexColumns: the index columns of the pivot

optional:
values: the values to be added

"""
    values = kwargs.pop('values', ['number_of_claims', 'policies_exposed'] +
                            map('expected_claim_qx{}vbt_by_policy'.format, [2008, 2015]))
    res = data.pivot_table(index = indexColumns
        , values=values
        , aggfunc=np.sum
        , dropna=True)
    res = res.reset_index()
    return res

def isPLT(s):
    """Tell whether is PLT, apply to """
    if 'yr anticipated' in s.anticipated_level_term_period:
        t = int(s['anticipated_level_term_period'].strip().split(' ')[0]);
        d = int(s['duration_group'].strip().replace('+','').split('-')[0])
        if t < d:
            return 'Y';
    return 'N';s

######################################################################################################################


def addPred(aGLMRun, pairsNewOffset, **kwargs):
    """Add the prediction as a column to data, in place.  
    Use the column name 'predname' for the new column.
arguments:
    aGLMRun: a 'glmrun' object, just need its 'fit' attribute
    parisNewOffset: list of pairs of column names: 
        [(new fitted column name, offset column name)
        , ... ]
        ... so can calculate dependency matrix and then just run with multiple offsets like amount and then count

kwargs:
    data      : dataframe to which to append the column, default is aGLMRun's data aGLMRun.fit.model.data.frame
"""
    fit  = aGLMRun.fit # for convenience
    data = kwargs.pop('data', fit.model.data.frame)
    dm   = patsy.dmatrix(fit.model.formula.split('~')[1], data=data, return_type='dataframe') #dm=design matrix
    # Must be careful to have columns in exog for all the parameters. 
    # Could use glmfit.params[dm.columns] if missing some.
    for newcol, offsetcol in pairsNewOffset:
        print(dt.now(), 'Predicting with offset '+offsetcol)
        data[newcol] = fit.model.predict(fit.params
                                      , exog   = dm
                                      , offset = fit.family.link(data[offsetcol]))
    print (dt.now(), '... predicted.')

def getKV(k):
    """Get key, value from label, value in the parameters"""
    if k=='Intercept':
        return ('Intercept','Intercept')
    m = re.match(r"^C\((.*)\)\[(.*.)]$", k) # has C in name to show is a category
    if m: #has C()
        m  = m.groups()
        m1 = list(m[1:])
        res = tuple([m[0].split(',')[0]]+m1) #remove "treatment" if there in 1st one
    else:
        m = re.match(r"^(.*)\[(.*.)]$", k) #lacks C showing cateogry

        if m:
            res = m.groups()
        else:
            return k; #just don't do anything
         #If field value is an integer then return it.
    if res[1][:2]=='T.':
        res = (res[0], res[1][2:]); # Strip the T. at the start of the value, I'm not sure why it's there.
    if re.match('^[\d]*$', res[1]) and res[1][0]!='0': # keep leading zeros if any, works better w/keys I have so far
        res = (res[0], int(res[1]))
    return res

def getKVs(k):
    """Get the KVs where multiple cross combinations are present."""
    if ':' in k: 
        return pd.Series({_k:_v for _k,_v in map(getKV, k.split(':'))})
    else:
        return getKV(k)
    
# from https://stackoverflow.com/questions/16705598/python-2-7-statsmodels-formatting-and-writing-summary-output
def results_summary_to_dataframe(results):
    '''This takes the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df

def fitGLM(data, formula, offsetcol):
    """run the fit, return it, set the offsetcol property of the fit.  Poisson with log link."""

    print(dt.now(), 'Fitting ...')
    fit = smf.glm(  data    = data
                  , formula = formula
                  , offset  = np.log(data[offsetcol])
                  , family  = sm.families.Poisson(link=sm.families.links.log())).fit()
    print(dt.now(), '... fit.')
    fit.offsetcol = offsetcol # to keep the name
    return fit
    
def offsetColumnName(fit):
    """Get the offset column name since the fit doesn't save it.  Assign it to attribute offsetColumnName of fit."""
    data = fit.model.data.frame # data in the model, for convenience
    # The numeric columns
    numericColumns = [_c 
                     for _c in data.columns
                     if 'int' in str(data[_c].dtype)
                          or 'float' in str(data[_c].dtype)]
    # Check the numeric columns to see whether they match the offset
    invoffset = fit.model.family.link.inverse(fit.model.offset) # will match an original column
    # Pick the column with the least root squared error from the offset column
    rse = pd.Series({_c: np.linalg.norm(data[_c]-invoffset) for _c in numericColumns})
    setattr(fit, 'offsetcol', rse[rse==rse.min()].index[0])
    return fit.offsetcol

class glmrun(object):
    """fit             "fit" instance, sved value of glm.fit()
    fit.model       is also "glm" above?
    fit.model.data.frame  The source dataframe, so not necessary to keep it separately
    fit.model.formula     The formula
        """

    def __init__(self, *args, **kwargs):
        """Pass either:
        1. fit object
        2. data, formula, offsetcol
        """
        if 'fit' in kwargs:
            self.fit = kwargs['fit']
            # set offset column name if need
            try:
                ocn = self.fit.offsetcol
            except:
                offsetColumnName(self.fit) # will set the attribute
        else: # assume has right bits
            self.fit = fitGLM(kwargs['data'], kwargs['formula'], kwargs['offsetcol'])
        
    def addPred(self, predname):
        addPred(self, predname) #addPred function takes this glmrun instance, uses its data and appends column to that.
        return self; # so can chain like d3

    def prettyParams(self):
        """Prettify the parameters: also add 1 where don't have one."""
        vals = {getKV(_k):_v for _k,_v in self.fit.params.to_dict().items()}
        s = pd.Series(vals)
        data = self.fit.model.data.frame; # for convenience
        #for each column, get any missing values - such as defaults, set their factor to 1 (not for Intercept - there's no corresponding column)
        for c in s.index.levels[0]: 
            if c!='Intercept':
                vals.update({(c, v):0. for v in list(set(data[c].value_counts().keys()).difference(set(s[c].index)))})
        return pd.Series(vals).sort_index()

    def writeFormula(self, doc):
        """Write the formula to the given document.  Return the document to allow chaining."""
        # FORMULA
        doc.add_paragraph("Formula for model {}, using offset column {}:".format(self.name, self.fit.offsetcol))
        r = doc.add_paragraph().add_run(self.fit.model.formula)
        r.font.name = 'Courier New'
        r.font.size = docx.shared.Pt(8)
        return doc
        
    def writeAppendixG(self, doc):
        """Write appendix g from the report into the docx."""
        doc.appendix_g_for_report(basis=self.name)

    def writeExhibit(self, doc, **kwargs):
        """
        NOTE: I'm not using it.
        Write an exhibit about a report to the document, number is from doc.n_tab.
Arguments:
    doc: a "mydoc" document object to which to write

kwargs: 
    show = True: whether to show in the IPython.notebook

Show: 
1. Formula
2. A/T vs factor (policies)
3. Graphs

A/Model table like exhibit G
"""
        show = kwargs.pop('show', True)

        # TITLE
        doc.add_paragraph('Model {}: Overview'.format(self.name), style='Caption').paragraph_format.keep_with_next = True

        # FORMULA
        self.writeFormula(doc)

        if show:
            dis.display(dis.HTML("<pre><span style='font-size:6;'>{}</span></pre>".format(self.fit.model.formula)))

            # Appendix G presentation but with this model
            self.writeAppendixG(doc); 

            # A/T AND MODEL FACTOR COMPARISON: NEEDS WORK FOR BIVARIATE SPLITS 
            self.compareFactorsAEPlot(doc=doc, show=show) # Makes the picture, saves to doc.

            """
            doc.add_df(pd.DataFrame({'model factor':self.prettyParams()
                                        , 'e(factor) = coefficient':np.exp(self.
                                        ())}).applymap('{:,.4f}'.format)
                          , caption='Model 0 multipilcative factors')
            """
        return doc # so can chain other work

    def writeSummaryTable(self, doc):
        """Write the summary table of glm in a new landscape section.
        pass: mydoc object"""    

        doc.landscapeSection(); # add a landscape section
        doc.add_paragraph('Model fit statistics: model {}'.format(self.name), style='Caption').paragraph_format.keep_with_next = True    
        smry = str(self.fit.summary().tables[1])
        r = doc.add_paragraph().add_run(smry)
        r.font.name = 'Courier New'
        r.font.size = docx.shared.Pt(6)
        # Display it in the notebook:
        dis.display(dis.HTML("<pre><span style='font-size:5pt;'>{}</span></pre>".format(smry)))

    # Utility function
    def getCoef(self, *args, **kwargs):
        """Get list of series of factors, index is values for given variable.
        Read from the glm output.
        The intercept is not returned."""

        # Get the components from the two sources.
        fitres = results_summary_to_dataframe(self.fit)['coeff'] # all results from the fit: coefficients only, so: a series

        # Get tuples in each row of exogenous variables, will be handy later: WILL CHOKE if one name is within another name!
        exog = set([tuple([c for c in sorted(self.fit.model.data.frame.columns) if c in en]) 
                    for en 
                    in self.fit.model.exog_names])
        exog = {deg:[f for f in exog if len(f)==deg]
                for deg in set([len(f) for f in exog]) if deg>0} #again, skip constant, deg=0

        # For each exogenous variable name set, get the coefficients as a series.
        coef = [] # will be list of series with index = the things by which splitting
        for deg in exog.keys():
            for names in exog[deg]:  #Get the fit results coefficients for just this variable
                # Get the coefficients for just this name and not for any others
                x = fitres[[i for i in fitres.index if (i.count(':') +1 == len(names)) and all([c in i for c in names])]].copy()
                # Change the index column names
                # Append to the coefficients
                #names = [getKV(k)[0] for k in x.keys()[0].split(':')]

                # trying a bunch of things that arent' working
                x = ({tuple([getKV(j)[1] for j in i.split(':')]):y
                     for i,y in x.to_dict().items()})
                ks,vs = [], [] # to keep them ordered the same
                for k,v in x.items():
                    ks.append(k)
                    vs.append(v)
                # Make the series
                s = pd.Series(vs, pd.MultiIndex.from_tuples(ks), name='coef')
                if len(names)>1:
                    s.index.names = names
                else:
                    s.index.name = names[0]            
                coef.append(s.sort_index())
        return coef
    
    """

    def compareFactorsAE(self, **kwargs):
        #Compare the factors and the A/E in the data
        #pp is the parameters (factors).
        #datsum is the data.
        pp = self.prettyParams()
        datsum = self.fit.model.data.frame
        lvls = [c for c in pp.index.levels[0] if c!='Intercept']
        
        # Graphing
        perRow = 2;  # how many graphs per row
        nrows = int(np.ceil(len(lvls)*1./perRow)) #leave off the intercept
        
        fig, ax = plt.subplots(nrows, int(perRow))
        fig.set_size_inches(4 * perRow, 3*nrows)
        fig.tight_layout()
        act = self.kwargs['formula'].split('~')[0].strip() #'number_of_claims' #Get from the formula.
        vbt = self.offsetcol # the expected, not generally an offset
        for _ax, _c in zip(ax.flatten(), lvls): # For each subplot, graph vs that index.
            tmp = datsum.pivot_table(index=_c, values=[act, vbt], aggfunc=np.sum);
            tmpdf = pd.DataFrame({
                ' factor':np.exp(pp[_c])
                , 'A/E':(tmp[act]/tmp[vbt])
                } );

            #Reorder for two oddballs
            if _c=='Duration_Band':
                tmpdf = tmpdf.loc[['(0, 5]', '(5, 10]', '(10, 15]', '(15, 20]', '(20, 25]']]
            elif _c=='Face_Amount_Band':
                tmpdf = tmpdf.loc[faceBands];

            tmpdf.plot(style=['ko-','bo-'], rot=90, ax=_ax, legend=True)
            # Show all indices if any are string (otherwise they're numeric)
            if reduce(operator.or_, map(lambda a: isinstance(a, str), tmpdf.index)):
                _ax.xaxis.set_ticks(np.arange(len(tmpdf.index)))
                _ax.xaxis.set_ticklabels(tmpdf.index)
        fig.suptitle="Factors vs Univariate A/{}".format(vbt)
        fig.tight_layout() # Not sure why did again doesn't seem to hurt
"""
    
    def compareFactorsAE(self, *args, **kwargs):
        """Compare factors and A/Model. NOTE: cross-combinations' univariate variables' factors are added in
        to the model factors can be compared to the A/E reflecting all that the model is doing.
    args: none
    
    kwargs:
        : none.
        """
        a,e = self.fit.model.formula.split('~')[0].strip(), self.fit.offsetcol # column names for actual and expected
        df  = self.fit.model.data.frame # The data
        coefOrig = self.getCoef(); # Coefficients as expressed by the results.

        # comp is a list of dataframes: index is categories, maybe multiindex; columns are a/table and factor.
        comp = []
        for sg in getSubgraphs(coefOrig): # must group the coefficients into groups that will be added together
            indexnames = list(reduce(set.union, map(set, sg))) # list if the names of fields in the index
            ae = df.pivot_table(index = indexnames, values = [a,e], aggfunc=np.sum)
            ae = ae[a] / ae[e]
            # Now: append the factors.  Add 0 in shape of ae first to get the full index so adding series will work.
            coefsum = addCoefs( [ae*0] + [c for c in coefOrig if c.index.names in sg] );
            tmp= pd.DataFrame({'Factor':np.exp(coefsum),'A/Table':ae}).sort_index()
            tmp.index.names = coefsum.index.names # they'll be the same.  For text categories the names didn't stay.
            comp.append(tmp)
        return comp    

    def compareFactorsAEPlot(self, *args, **kwargs):
        """Shows the comparison plots of a/e vs factors for this model.   Written to document if passed, shown in notebook if wanted.

    kwargs:
        doc: the mydoc document to which to write
        show_analysis: False
        """
        comp = self.compareFactorsAE()
        show_analysis = kwargs.pop('show_analysis', False)

        if kwargs.has_key('doc'):
            doc = kwargs.pop('doc');
            for c in comp:
                # Save to doc;
                compPlot(c);
                doc.add_fig('Model {}, Factors by {} vs Actual/VBT2015'.format(self.name, c.index.name))
                if show_analysis:
                    # Skip the caption argument.  The title above should suffice.
                    doc.add_df(self.factorAnalysisExhibit(c.index.name).applymap('{:,.2f}'.format), size=6)
        else:
            for c in comp:
                compPlot(c)
        return comp;


    def showFactorsVsTableBetweenCatgories(self, *args, **kwargs):
        """Plot ratios of combinations of factors, allowing visualization of differences where they're off between actual/table between categories and the factors for those categories.

kwargs:
    plot=True: if should plot
    doc: mydoc object if will plot and want to add a figure
    
returns:
    dataframe with index showing pairs for further work

        """
        t = self

        # Old version, worked well with no cross combinations
        """c = pd.concat(t.compareFactorsAE(), axis=0, keys = [_df.index.name for _df in t.compareFactorsAE()])
        d = {(c.iloc[a].name[0],c.iloc[b].name[0],c.iloc[a].name[1],c.iloc[b].name[1]): 
                 (c.iloc[a]/c.iloc[b]) # the ratio: of both a/t and factor at row a over those at row b
                for a,b in itertools.combinations(range(len(c)), 2)
                if c.iloc[a].name[0]==c.iloc[b].name[0] # only  compare within the same variables
            }
        res = pd.DataFrame(d).T
        res.index.names = ['num_k','den_k', 'num_v', 'den_v']

        """
        res = {( str(c.index.to_native_types()[a])
                ,str(c.index.to_native_types()[b])):(c.iloc[a]/c.iloc[b])
                 for c in t.compareFactorsAE()
                 for a,b in itertools.combinations(range(len(c)),2)
                                      } # all combinations between rows
        res =  pd.DataFrame(res).T
        res.index.names = ['numerator','deonominator']
        
        plot = kwargs.get('plot', True)
        if plot or kwargs.has_key('doc'):
            # Plot the pairs
            fig, ax = plt.subplots(1,1)
            fig.set_size_inches(5,5)

            mx = res.values.max()*1.1
            res.plot.scatter(x=0,y=1, xlim=(0,mx), ylim=(0,mx), s=2, ax=ax, color='k', alpha=0.4)

            plt.plot([0,mx], [0,mx], 'g') # line for equality
            # Red zones: not even same directional relationship. Near red zones also not great.
            ax.add_patch(mpl.patches.Rectangle((1, 0), mx, 1, angle=0.0, alpha=0.2, color='r'))
            ax.add_patch(mpl.patches.Rectangle((0, 1), 1, mx, angle=0.0, alpha=0.2, color='r'))

            ax.set_xlabel('Ratio of A/Table ratios between categories')
            ax.set_ylabel('Ratio of model factors between categories')
            ax.set_ylim(0,2);
            ax.set_xlim(0,2);
            
            if not kwargs.has_key('doc'):
                ax.set_title('Relationships between Categories\nOf Raw A/Table Ratios and Categorical Factors\nFrom model '+self.name)
            else:
                doc = kwargs['doc']
                doc.add_fig('Model {} Relationships between Categories of Raw A/Table Ratios and Categorical Factors'.format(self.name))

        return res
    

    def compFactorDist(self, indexColumn, forEach, plot=False, *args, **kwargs):
        """Return: distribution of offset column (exposure, expected, etc)  across index column values for each separate value of field "forEach"
        
        kwargs:
            data: default is GLM's data
            ofColumn: column of which to take the distribution, default is the offset column
        
        Example: where t is an instance:  plot ia_band1 factors on secondary y axis, 
                    and distribution of each insurance_plan by ia_band1 
            
            t.compFactorDist('ia_band1', 'insurance_plan').plot(secondary_y='Factor')
            
        
        """
        data = kwargs.pop('data', self.fit.model.data.frame)
        ofColumn  = kwargs.pop('ofColumn', self.fit.offsetcol)
        return compFactorDist(self, data, ofColumn, indexColumn, forEach, plot);

    def compFactorAvg(self, forEach, *args, **kwargs):
        """Compare average factors weighted by the fit's offset column for different splits in forEach.
        arguments:
        kwargs:
            data, default is the GLM's data
            ofColumn: column by which to weight the factors to get the average, such as exposure or expected claims; 
                        default is model's offset column"""
        data     = kwargs.pop('data', self.fit.model.data.frame)
        ofColumn = kwargs.pop('ofColumn', self.fit.offsetcol)

        # To avoid repeated calls
        prepar = self.prettyParams()
        
        # List the levels in the model to compare. All of them but the intercept and the one for which we'll show
        # the factor, since for that one the average is just the factor.
        
        lvls = list(set(prepar.index.levels[0]).difference(['Intercept', forEach]))
        
        #function to get weighted average of factors.
        # The dataframe passed hasthe factor as the 1st column, and is output of compFactorDist.
        avgFactor = lambda df: df[[c for c in df.columns if c != 'Factor']].multiply(df['Factor'], axis=0).sum() 
        res = pd.DataFrame({lvl:avgFactor(self.compFactorDist(lvl, forEach, plot=False, data=data, ofColumn=ofColumn)) 
                            for lvl in lvls})  # get the average factor for each lvl
        return pd.concat([ # Get the factors themselves
                          #self.fit.family.link.inverse(pd.DataFrame({forEach:prepar[forEach][list(set(data[forEach]))]}))
                           self.fit.family.link.inverse(pd.DataFrame({forEach:prepar[forEach]}))
                            #add column for factors for thing we're splitting by
                         , res], axis=1).T #append factors for thing we're splitting by and return.
    
    
    
    def factorAnalysisExhibit(self, forEach):
        """Show across categories in forEach: 
        
        A/Table
        Factor
        Average of all other factors
        Approximate A/Table (product of all factors or average thereof) 
       
       arguments:
           forEach: should be a category in the model
           
       ... Compare factors for a split with the average factor in the applicable other categories and a/table"""
        cf = {_df.index.name:_df for _df in self.compareFactorsAE()}[forEach].T
        cfa = self.compFactorAvg(forEach) # the average factor: must exclude category forEach from this becasue is shown
                 # as factor already
        # Show intercept (overall factor) and take out the one that we're splitting by since we show it already
        cfa = pd.concat([
            pd.DataFrame(self.fit.model.family.link.inverse(self.prettyParams()[('Intercept', 'Intercept')])
                             , index=['Overall']
                         , columns=cfa.columns)
            , cfa.loc[cfa.index!= forEach]], axis=0            
            )
        # datafame of a/t and the estimate
        atdf = pd.DataFrame({'Observed':cf.loc['A/Table']
                             , 'Approximated':cfa.product(axis=0) * cf.loc['Factor']})[
                            ['Observed','Approximated'] # to reorder
                        ].T
        res = pd.concat([atdf # The A/T that we'd like to explain
                         , pd.DataFrame({forEach:cf.loc['Factor']}).T # the factors themselves
                         , cfa
                         ]
                         , keys=['A/VBT15', 'Factor','Avg factors']) # The overall factor (intercept) will show as avg, is OK
        res.index.names = ['',''] # so don't show "None","None" when writing out
        return res

                              
def compFactorDist(aGLMRun, data, ofColumn, indexColumn, forEach, plot=True):
    """Align factor with distribution between categories specified by indexColumn,
    for item in ofColumn (such as expected claims or exposure),
    splitting distribution in forEach, 
    really only works for log link.
    
    Returns: dataframe, split by indexColumn down rows (dataframe index); 
        Columns: 'Factor' for factor of indexColumn, 
                then a column for each elelement of column forEach (such as for each insurance_plan)
    """
    dist = getDist(data, ofColumn, indexColumn, forEach)
    # Get the factor for indexColumn
    fact = pd.DataFrame({'Factor':aGLMRun.fit.family.link.inverse(aGLMRun.prettyParams().loc[indexColumn])})
    if plot:
        fig, ax = plt.subplots(1)
        dist.plot(kind='bar', ax=ax, rot=90, title='Distribution of {}'.format(ofColumn))
        fact.plot(secondary_y='Factor', ax=ax, rot=90)
        ax.legend(loc='upper left', bbox_to_anchor=(1.1,1));
    return pd.concat([fact, dist], axis=1).fillna(0)



def getDist(data, ofColumn, indexColumn, forEach):
    """Get distribution of value in ofColumn between categories of indexColumn, splitting by forEach (forEach is columns in result set)"""
    tmp = data.pivot_table(index=indexColumn, columns=forEach, values=ofColumn, aggfunc=np.sum)
    return tmp.div(tmp.sum(), axis=1)



def compareAE(data, index, columns, *args, **kwargs):
    """Show a to e by index and columns for quick inspection of results.
    Shows: rows: top: count, bottom: amount
        Columns: different bases: vbt15, g_count, g_amount, (then g_tree?)
    """
    bases   = ['qx2015vbt','g_count','g_amount']
    metrics = ['policy', 'amount']
    numerator = {'policy':'number_of_deaths', 'amount':'death_claim_amount'} # because source has nonstd names
    pt = data.pivot_table(index=index, columns=columns
                          , values=(numerator.values()  +
                                    ['expected_death_{}_by_{}'.format(basis, metric)
                                     for basis, metric
                                     in itertools.product(bases, metrics)])
                          , aggfunc=np.sum)
    # For conctenating the dataframes:
    res = {(metric,basis):(pt[numerator[metric]]/pt['expected_death_{}_by_{}'.format(basis, metric)])
                    for metric, basis
                    in itertools.product(metrics, bases)}
    if kwargs.pop('plot',True):
        fig, ax = plt.subplots(len(metrics), len(bases))
        fig.set_size_inches(4*ax.shape[1], 3*ax.shape[0])
        fig.tight_layout()
        for _ax, _key in zip(ax.flatten(), itertools.product(metrics, bases)):
            showLegend =  ( _key==(metrics[0], bases[-1]))
            res[_key].plot(rot=90, ax=_ax, legend=showLegend # top right only
                           , title='a/{} by {}'.format(_key[1], _key[0]), ylim=(0.5, 2)
                           , style=[c+s for s, c in  itertools.product( ['-',':','--', '-.'], 'kbrg') ])
            _ax.xaxis.set_ticks(np.arange(len(res[_key].index)))
            _ax.xaxis.set_ticklabels(res[_key].index)
            if showLegend:
               _ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        plt.tight_layout() # need this too?
    return res

