# Module for saving functions for use between notebooks
# number of values in each key

try:
    univariateSummaries = pd.read_pickle("../dat/univariateSummaries.pkl")
except: # following added 2019 for running this from a higher directory.
    try:
        univariateSummaries = pd.read_pickle('20180606_Analyses/dat/univariateSummaries.pkl');
    except:
        print('Failed to load univariateSummaries from dat/univariateSummaries.pkl')

keynums = pd.Series({k:len(univariateSummaries.loc[k]) for k in univariateSummaries.index.levels[0]});


def plot_sysplit_sy_lines( sysplit, c):
    """Plot studyyears as lines across categories in c"""
    
    fig, ax = plt.subplots(2, 2)
    
    for j in [0,1]:
        ax[0, j].get_shared_x_axes().join(ax[0, j], ax[1, j])
    
    fig.set_size_inches(2 * 4, 2 * 3)
    
    for _ax, _v in zip(ax.flatten()
                       , ['amount_exposed', 'a/2015vbt by amount'
                          , 'policies_exposed', 'a/2015vbt by policy']):
        df = sysplit.loc[c, _v].unstack() 
        df.index.name = 'observation year'
        if not c in ['ia5', 'aa5', 'duration']:
            df.columns = map(str, df.columns)
        df.columns.name = c
        df = df.sort_index(axis=1)
        
        cols = reversed([(c,c,c) for c in  np.arange(0, 1, 1./len(df))]) # colors
        ylim = {True:0, False:(0.75, 1.25)}['exposed' in _v]
        df.T.plot(title=_v, style='-', ylim=ylim, ax=_ax, legend=False, rot=90, color=cols, alpha = 0.5)

    for _ax in ax[:, 1]: # pcts
        _ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter('{:,.0%}'.format))

    # set x axis labels if nonnumeric
    if not c in ['ia5', 'aa5', 'duration']:
        for _a in ax[1]:
            _a.set_xticks(range(len(df.columns)))
            _a.set_xticklabels(df.columns)

    fig.tight_layout()
    fig.savefig('graphs/1_{}.svg'.format(c))
    fig.savefig('graphs/1_{}.png'.format(c), dpi=300)
    return df
    
def plot_sysplit_x_axis(sysplit, c):
    """plot splits with study years across x axis, not categories"""
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(2 * 4, 2 * 3)
    for _ax, _v in zip(ax[:, 0], ['amount_exposed',  'policies_exposed']):
        df = sysplit.loc[c, _v].unstack()
        df.index.name = 'observation_year'
        df.columns.name = c
        df['Total'] = df.sum(axis=1)
        
        df.plot(title=_v, style='.-', ylim=0, ax=_ax)
    for _ax, _v in zip(ax[:, 1], ['a/2015vbt by amount',  'a/2015vbt by policy']):
        df = sysplit.loc[c, _v].unstack()
        df.index.name = 'observation_year'
        df.columns.name = c
        df.plot(title=_v, style='.-', ylim=(0.75, 1.25), ax=_ax)
    # Pcts on the R for a/table
    for _ax in ax[:, 1]: # pcts
        _ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter('{:,.0%}'.format))

        
    fig.tight_layout()
    fig.savefig('graphs/1_{}.png'.format(c), dpi=300)
    fig.savefig('graphs/1_{}.svg'.format(c))
    return df

###########################################
# Rollforward work

def rfw(dIn, yIn, steps =3, metric='amt'):
    """Rollforward to year y using data d in given steps, all possible orders 
steps: number of steps
yIn: year into which go (from yIn-1 to yIn)
metric: 'amt' or 'cou' for amount or count

Here: t = table amount, x = exposure, a = actual amount:

At each step show a/vbt15 in total.
-2. prior year
-1. Remove exposures missing from year yIn
All orders are tested for each update.
    If 3 steps: 0, 1, 2
        Change each one of x, a/t, and t/x (note: t/x s/n move since is sufficiently granular), in turn
    
   if 2 steps: steps 0, 1:  
       Change each of x, a/x.  t changes when x changes. +
3. Add in exposure, expecteds, actuals present in year yIn but not in yIn-1
4. Show currrent year as a check, should match prior step"""
    
    # column names in the original
    x, t, a  = ('_'.join([metric, n]) for n in ['xps','vbt15','act'])
    w  = yIn-1 # prior year
    da = dIn.copy().fillna(0) # all data

    # get the ratios
    for z in [w,yIn]:
        for _s in 'atx':
            da[(_s, z)]   = da[(eval(_s), z)]
        da[('a/t', z)] = da[(a, z)] / da[(t, z)]
        da[('t/x', z)] = da[(t, z)] / da[(x, z)]
        da[('x'  , z)] = da[(x, z)]

    db = da[(da[(x, w)]>0) & (da[(x, yIn)]>0)] # are exposures in both years

    fullres = {}
    
    for s in itertools.permutations(range(steps)): # the permutation of the order
        res = {}

        y = [w, w, w]  # years used for each step in the ratio
        # starting point: same for each permutation, is redundant, yet we live
        for i, d in zip([-2, -1], [da, db]): # all data at prior year, then data in both yrs at prior year
            res[i] =(  (d[('a/t', y[0])] * d[('t/x',  y[1])] * d[('x', y[2])]).sum() 
                     / (                    d[('t/x',  y[1])] * d[('x', y[2])]).sum())

        
        d = db # use data has exposure for both years
        # update components in order
        # s is some permutation of [0, 1] or [0, 1, 2]
        for i, j in enumerate(s): 
            y[j] += 1 
            if j==1 and steps==2: # must also increment y[2], move the tabular expected and exposure together
                y[2] += 1

            res[i] = (  (d[('a/t', y[0])] * d[('t/x',  y[1])] * d[('x', y[2])]).sum() 
                   / (                      d[('t/x',  y[1])] * d[('x', y[2])]).sum())

        # All data in new year: step 2 or 3 respectively
        i += 1
        d = da
        res[i] =  (  (d[('a/t', y[0])] * d[('t/x',  y[1])] * d[('x', y[2])]).sum() 
                / (                      d[('t/x',  y[1])] * d[('x', y[2])]).sum())
        i += 1
        # check that end up in right place
        res[i] = d[(a, yIn)].sum() / d[(t, yIn)].sum()
        if steps==3:
            fullres[tuple([['a/t', 't/x', 'x'][_s] for _s in s])] = res
        else: # steps==2; update t, x both simultaneously
            fullres[tuple([['a/t', 't, x'][_s] for _s in s])] = res
    ret = pd.DataFrame(fullres)
    ret.columns.names = [1,2,3][:steps]
    return ret


def large_graph(d, col, title):
    """Make a large graph focused on a/e with the given split."""

    fs = 14 # baseline font size to use
    
    pt = d.pivot_table(index='observation_year'
                       , columns=col
                       , values=['death_claim_amount', 'expected_death_qx2015vbt_by_amount']
                       , aggfunc=np.sum, margins=True, margins_name='Total').iloc[:-1] # drop total line at end for years
    
    # special formatting for total line
    ae = pt['death_claim_amount']/pt['expected_death_qx2015vbt_by_amount']
    
    cpref = col.split('_')[0]
    ae.columns.name = {'insurance':'Plan', 'face':'Face amount', 'dur':'Duration', 'ia':'Issue age'}.get(cpref,cpref)
    
    styles = [''.join([x[1],x[2],x[0]]) for x in itertools.product(['-',':'], 'bgrm', ['x', '+', '^', '.'])]
    styles = styles[:len(ae.columns)-1]+['ko-'] # black for total
    ax = ae.plot(  figsize=(10,6)
                 , style=styles
                 , ylim=(0.75, 1.25)
                 , ms=10
                 , fontsize = fs
                 , title=title)
    
    ax.set_yticks(np.arange(0.75, 1.25, 0.05))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:,.0%}'.format))
    
    #plt.setp(ax.get_xticklabels(), fontsize=fs)
    #plt.setp(ax.get_yticklabels(), fontsize=fs)
    plt.title(title, fontsize=fs+2)
    
    plt.ylabel('A/2015VBT by Amount', fontsize=fs)
    
    plt.grid(b=True, axis='y')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1),  title=ae.columns.name
               , prop={'size': fs})
    plt.tight_layout()
    plt.savefig('graphs/2_{}_{}.png'.format(title, col), dpi=300)
    return ae


# Graph Rollforward
def graphRollforward(d, type=0, metric='amt', **kwargs):
    """type:
    0: 2 steps, skipping the data change
    1: 2 steps, showing the data change
    2: 3 steps, showing the data change"""
    steps = {0:2, 1:2, 2:3}[type]
    
    ymin, ymax = d.columns.to_frame()['observation_year'].min(), d.columns.to_frame()['observation_year'].max()
    
    # pull togethereach year of the Rollforward.
    r = pd.concat([rfw(d, y, steps=steps, metric=metric)
                   for y in range(ymin+1, ymax+1)]
                  , keys=range(ymin+1, ymax+1)).sort_index(axis=0).sort_index(axis=1)

    if type==0:
        # Just get the steps we need: start at -2, end at 3, and 0 is in between.
        # Note: steps are strings when returned by to_native_types.
        stepmap = {-2:-1, 0:-0.5, 3: 0.} # map steps to locations offset from year
        r = pd.DataFrame({(int(i)*1.+ stepmap[int(j)]):
                              r.loc[(int(i), int(j))] 
                          for i,j in r.index.to_native_types()
                          if int(j) in stepmap.keys()
                                                 }).T
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(8, 6)
        colors = ['b', 'r']
        for i in np.arange(ymin, ymax):
            xs = [[i, i+0.5], [i+0.5, i+1]]
            
            for j in [0,1]:  # for 1st step then 2nd step
                # Plot change in q, just read the dataframe to make sure these are right.
                plt.plot(xs[j], list(r.loc(axis=0)[xs[j]].iloc[:, j]), colors[0]+'-')
            
                # Plot change in expected vbt
                plt.plot(xs[j], list(r.loc(axis=0)[xs[j]].iloc[:, 1-j]), colors[1]+'-')
            
        import matplotlib.lines as mlines
        # Draw the legend
        plt.legend(handles=[mlines.Line2D([], [], color=colors[0], label='Update actual/vbt')
                            ,mlines.Line2D([], [], color=colors[1], label='Update exposure and 15VBT expecteds') 
                           ], title='Item changed')
        ax.set_xlim(ymin-1, ymax+1)
        
            
    elif type in [1, 2]:
        # turn index from year and step number to fractions between years for plotting.
        imin, imax = min(r.index.levels[1]), max(r.index.levels[1])
        r = r.loc(axis=0)[:, range(imin, imax)] # leave of last one in each since is just repetition (imax)
        r.index = [np.float(y) + (np.float(s)-(imax-1))/((imax-1)-imin) for y, s in r.index.to_native_types()]  # show as steps between years

        # Plot the paths. 
        ax = r.plot(style=['.-', '.-'], figsize=(10,6), fontsize=14)

    # Common to all types:
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:,.0%}'.format))
    
    # Plot text labels for the results at each year
    for x in range(ymin, ymax+1):
        try: # might be 2 rows ( a dataframe ) returned since intermediate years are doubled
            y = r.loc[x].iloc[0,0] # the 1st value, doesn't matter, they're the same
        except:
            y = r.loc[x].iloc[0] 
        plt.text(x-0.1, y+0.002, '{:,.1%}'.format(y))
    
    # plot large dots of the actual points
    r.loc[range(ymin, ymax+1)].iloc[:, 0].plot(style='ko:', ax=ax)    
    plt.grid()

    if 'name' in kwargs:
        plt.title(kwargs['name'], fontsize=14);
        plt.savefig('graphs/3_{}.png'.format(kwargs['name']), dpi=300)

    return r
