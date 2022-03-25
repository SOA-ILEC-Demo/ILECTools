"""Code used in the Rollforward paper

cols_valact:
valcols: value column list
tabkeys: key value columns for mortality rates: qx should only vary by these
class Rollforward: for working with rollforwards, error metrics etc

save_data: FIX - MAKE THIS: saves the pkls, makes from the downloaded files

"""

import matplotlib as mpl, matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from IPython.display import HTML


# columns in valact: for the detailed split for decomp?
cols_valact = ('preferred_indicator,gender,smoker_status,face_amount_band,number_of_preferred_classes,preferred_class,insurance_plan'.split(',')
               +'issue_age,duration,soa_post_level_term_indicator'.split(','))
# columns in valact for the Rollforward
cols_valact_rf= ['issue_age',
 'soa_post_level_term_indicator',
 'insurance_plan',
 'gender',
 'duration',
 'smoker_status']


class Rollforward():
    """
Attributes:
        
        title: a title of the set
        dat: data passed
        keys: the keys over which aggregte
        
        pt: pivot of data passed used in other items
        
        R, dR : A/Table by year, forward difference
        w, dw : weight of tabular rates w/in each year, forward difference
        
        r, dr : a/table, forward difference (no change if one of ratios is null e.g. if denominator=tabular rate=0)
        r_bf: same as r: is r filling nulls by going backward then forward
        r_fb: is r filling nulls by going forward then backward

Functions - main ones
    R_vs_mean_weights
    diagnostics
    show_dr_dw
    show
    waterfall
    
    """
    
    def __init__(self, title, dat, keys,  **kwargs):
        """Pass 
        dat: dataframe, data to use: must have observation_year, amount_actual, amount_2015vbt, amount_exposure
        keys: the column names by which to split
    optional kwargs:
        title='Untitled'
        metric='amount': the metric, either 'amount' or  'policy'
        
        """
        
        metric = kwargs.pop('metric', 'amount')
        self.metric = metric
        # Make sure observation_year is there and in the innermost level
        a,t,x = ['actual','2015vbt','exposure'] # actuals, tabulars, exposure
        # summarize at the desired level of granularity within each year
        pt = dat.pivot_table(index = list(set(keys).difference(['observation_year'])) + ['observation_year']
                            , values=['_'.join([metric,i]) for i in (a,t,x)]
                            , aggfunc=np.sum, fill_value=0)\
            .rename(columns=lambda c: c.split('_')[-1], level=0) # to remove metric from column name
        # Rename the a, t, x to exclude the metric
        pt = pt[pt[t]>0] # exclude a few problematic cells with actuals and no expecteds: ust in case
        # set the ratio
        pt['r'] = pt[a] / pt[t] # don't fill with na yet
        pt = pt.unstack() # to get years across columns in innermost layer
        
        # Total ratios: real values, for validation
        pts = pt.fillna(0).sum().drop('r') # r makes no sense here (adding ratios)
        R = pts[a] / pts[t]  
        dR = R.diff().shift(-1) # shift back to get forward difference from backward difference
        
        # set r, dr
        r = pt['r'] # for convenience
        dr = r.diff(axis=1).shift(-1, axis=1).fillna(0) # if one of r_y or r_{y+1} is n/a then assume no change: fill with 0
        # r filling nulls going back and then forward for any in latest years
        r_bf = r.T.bfill().ffill().fillna(0).T # can fill back and fwd adn with 0 and s/n impact r.
        # r filling nulls going forward and then backward for any in latest years
        r_fb = r.T.ffill().bfill().fillna(0).T
        r = r_bf
        
        # set w, dw
        w = pt[t].fillna(0).div(pts[t]) # share of the table
        dw = w.diff(axis=1).shift(-1, axis=1) # difference in the mix from that year to the next. Will not have na since has 0s in the amounts
    
        for k in 'title,dat,keys,pt,R,dR,r,r_bf,r_fb,dr,w,dw'.split(','):
            setattr(self, k, eval(k))

    def show_dr_dw(self):
        """Show the 4 components of the product of differences: whether each is up or down"""
        dr,dw = self.dr, self.dw

        drdw = dr*dw
        
        # split up dr dw
        drp = dr>0
        dwp = dw>0
        drn = ~drp
        dwn = ~dwp

        
        tmp = pd.DataFrame({'dr+ dw+':(drdw*drp*dwp).sum()
             ,'dr- dw-':(drdw*drn*dwn).sum()
             ,'dr+ dw-':(drdw*drp*dwn).sum()
             ,'dr- dw+':(drdw*drn*dwp).sum()
             }).iloc[:-1]
        tmp['Total'] = tmp.T.sum()
        ax = tmp.plot(style=['g-','r:','g:','r-', 'b-'], alpha=0.5, xlim=(min(tmp.index)-1, max(tmp.index)+1), title=self.title)
        #.style.format('{:,.2%}')
        for i in tmp.index.to_list():
            ax.text(i, tmp.loc[i, 'Total'], '{:,.1%}'.format(tmp.loc[i, 'Total']))

        ax.axhline(0, color='k', alpha=0.5)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:,.0%}'.format))
        plt.legend(bbox_to_anchor=(1,1), loc='upper left');

    def show(self):
        R, r, dr, w, dw = tuple(getattr(self, k) for k in 'R,r,dr,w,dw'.split(',')) # for convenience
        tmp = pd.DataFrame({'R':R, 'r dw':(r*dw).sum(), 'dr w':(dr*w).sum(), 'dr dw':(dr*dw).sum()})
        tmp['Total'] = tmp.sum(axis=1)
        return tmp.style.format('{:,.1%}').set_caption(self.title + ' Rollforward')

    
    def diagnostics(self):
        """Various diagnostics of the summary such as records with 0 weight between years"""
        # where weight is zero in this or next year
        w0 = ((self.w==0) | (self.w==0).shift(-1, axis=1))

        return HTML("""
    Summary: {:,.0f} records
    <hr/>
    <table><tr><td>{}</td><td>{}</td></tr></table>
    """.format(
        len(self.pt)
        , pd.DataFrame({'# recs':(self.dr.fillna(0)==0).sum()
                 ,'% recs':((self.dr==0)*1.).sum()/self.dr.shape[0]
                 ,'weight':((self.dr==0)*self.dw).sum()
                 }).style.format({'# recs':'{:,.0f}', '% recs':'{:,.1%}', 'weight':'{:,.1%}'}
                                 ).set_caption("where dr=0:").to_html()
        , pd.DataFrame({'# recs':w0.sum()
                         ,'% recs':1.*w0.sum()/self.dr.shape[0]
                         ,'weight':(w0*self.dw).sum()
                                 }).style.format({'# recs':'{:,.0f}', '% recs':'{:,.1%}', 'weight':'{:,.1%}'}
                                                 ).set_caption("where w=0 in this year or next:").to_html()

        ))
    
    def __repr__(self):
        return '{}, split by {}'.format(self.title, list(self.keys))

    def R_mean_weights(self, filldir=0):
        """Return R for each year but using the a/t for that year and the weights for each cell averaged across years.
        
Parameters to deal with missing r (actual / table in a cell in a year):
    filldir:  0 to fill values backward in that cell and then forward
              1 to fill values forward over subsequent nulls in that cell and then backward 
        """
        # value for any remaining n/a values in r, would make no difference: could only be some if weight is 0 in all years.
        fillna=0.
        w_mean = self.w.mean(axis=1)
        if filldir==0:
            _r = self.r_bf
        else:
            _r = self.r_fb
        
        return _r.mul(w_mean, axis=0).sum()
    
        pd.DataFrame({'R':self.r.R
         , 'R with avg mix and r bfill ffill':r.r.mul(w_mean, axis=0).sum() 
            # where was no exposure for r in that year in that cell, it's 0.  
            # Prob not invalid, they're small cells and would likely have had nothing.
         ,'R with avg mix and r ffill bfill fillna -1000':(r.pt['act']/r.pt['2015vbt'].fillna(0))\
                      .ffill(axis=1).bfill(axis=1).fillna(-1000).mul(w_mean, axis=0).sum()
                      }).plot(style=['-', 'k--', 'r--', 'go', 'rx'])


    def R_vs_mean_weights(self, plot=True, **kwargs ):
        """Return the overall R = overal A/overall T vs recomputed R for each year using the weights averaged across each year, both filling forward and backward.
        plot=True: whether to plot
        optional kwargs:
            ax: axes on which to plot, ignored if plot=False"""
        res = pd.DataFrame({'R':self.R
                             , 'R with avg mix, filling *r* forward then backward':self.R_mean_weights(filldir=1)
                             , 'R with avg mix, filling *r* backward then forward':self.R_mean_weights(filldir=0)
                            })
        if plot:
            if 'ax' in kwargs:
                _a = kwargs['ax']
            else:
                _, _a = plt.subplots(1,1)
            res.plot(ax=_a)            
            _a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
            _a.grid(which='major', ls='-', c='k', axis='y')
            _a.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
            _a.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))
            _a.grid(which='major', ls='--', axis='x')
            _a.grid(which='minor', ls='--', axis='y')
            
        return res
        

    
    def waterfall(self, plot=True, **kwargs):
        """Return the waterfall dataframe.  Plot by default, pass plot=False not to plot.
        
        pass optional ax=(axes object) to specify axes on which to plot."""
        #rf = components.data # since a styler was passed
        #rfc = components.data.cumsum(axis=1).iloc[:, :-1]
        
        # Get cumulative sum
        rf = pd.DataFrame({'r w':(self.r * self.w).sum()
             , 'dr w':(self.dr * self.w ).sum()
             , 'r dw':(self.r * self.dw ).sum()
             , 'dr dw':(self.dr * self.dw ).sum()
            } )[['r w',  'r dw', 'dr w', 'dr dw']]
        
        if plot:
            rfc = rf.cumsum(axis=1)[:-1]
            if 'ax' in kwargs:
                ax = kwargs.pop('ax')
                fig = ax.figure
            else:
                fig, ax = plt.subplots(1,1)
            R = self.R
            R.plot(style='ko-', ax=ax, grid=True, alpha = 0.5)

            # Plot text labels for the results at each year
            for yr, ae in R.to_dict().items():
                ax.text(yr-0.1, ae+0.005, '{:,.1%}'.format(ae), rotation=45)

            w = 1./3 * 2./3
            boxes = [[], [], []] # boxes for the 3 steps

            for yr in rfc.index:
                for i, b in zip(range(3), boxes): # for each column
                    # draw line of step
                    v0 = rfc.loc[yr].iloc[i]
                    v1 = rfc.loc[yr].iloc[i+1]
                    ax.plot([yr+i/3., yr+(i+1)/ 3.], [v0]*2, 'k-')
                    b.append(mpl.patches.Rectangle((yr+(1+i)/3. - 0.5*w, v0), w, v1-v0))

            colors = 'gbr'
            # Create patch collection with specified colour/alpha
            for b, c in zip(boxes, colors):
                pc = mpl.collections.PatchCollection(b, facecolor=c, alpha=0.4, edgecolor='None')
                ax.add_collection(pc)

            # add the legend
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color=c, lw=4, alpha=0.4) for c in colors]
            ax.legend(custom_lines, ['change from weights', 'change from rates', "interaction"])        

            ax.set_xlim((R.index[0]-0.5, R.index[-1]+0.5)) 
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:,.0%}'.format))
            fig.set_dpi(150)

        rf['Total'] = rf.sum(axis=1)
        return rf.iloc[:-1].rename_axis(index='y').rename(columns={'r w':'$R_y$'
                                               ,'r dw': r'$\v{r}_y\cdot\Delta\v{w}_y$' # r dw
                                   , 'dr w':r'$\Delta\v{r}_y\cdot\v{w}_y$' # dr w
                                   , 'dr dw':r'$\Delta\v{r}_y\cdot\Delta\v{w}_y$' 
                                   , 'Total':'$R_{y+1}$'
                                                                  })\
                .style.format('{:,.1%}')

    
    def weight_angles(self):
        """Show angles in degrees between weights of pairs of years."""
        w_norm = np.linalg.norm(self.w, axis=0)
        # Must clip to be between -1 and 1s
        return (np.arccos(np.clip(self.w.T.dot(self.w).div(w_norm, axis=1).div(w_norm, axis=0)
               , -1, 1)) / 2 / np.pi * 360 ).style.format('{:,.1f}').set_caption('Angles between years in degrees')
    
    
    
    def granularity_check(self, y, scale=0.01, item=['r']):
        """
    Distribution of an item  by cell for year y.  If cells are small then will be highly skewed with much exposure in cells with no claims.
    Default item is ['r'], r=a/t; may choose r, dr, w, dw
    Default scale is 0.01
    returns exposure by bucket at given scale.  e.g. at default scale 0.01, buckets are
        0.00: [0.00, 0.01)
        0.01: [0.01, 0.02)

    ... etc ...
        """
        wrk = pd.concat( [self.r,self.w,self.dr,self.dw]
                    , keys='r,w,dr,dw'.split(','), axis=1).swaplevel(0,1,axis=1).sort_index(axis=1)[y]
        binnames = []
        for i in item:
            n = 'bin_'+i
            binnames.append(n)
            wrk[n] = wrk[i].map(lambda x: int(x/scale)*scale)
        return  wrk.pivot_table(index=binnames, values='w', aggfunc=np.sum)
