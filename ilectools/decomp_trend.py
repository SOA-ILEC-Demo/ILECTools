"""Used in 201905_Decomposition_Trend.ipynb
Development of presentation functions

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# not used - is an older technique
def fill_between(ax, a, da):
    """red when go down frmo a to a + da; blue when go up from a to b.
    NOT USED in final - I went to the bar graphs."""
    x = range(len(a))
    ax.fill_between(x, a.values, (a+da.apply(lambda x: min(x, 0))).values, color=(1, 0.7, 0.7), linewidth=0) # going down
    ax.fill_between(x, a.values, (a+da.apply(lambda x: max(x, 0))).values, color=(0.7, 0.7, 1), linewidth=0) # blue going up
    ax.plot(x, a.values, color='k', linewidth=0.5)  # the mean, same in all as a reference point
    ax.plot(x, (a+da).values, color='k', linewidth=1)  # the result
    ax.plot(x, 0*a.values, color=(0.5, 0.5, 0.5))  # 0 on the y axis
    # Prettification
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    
def get_comp(x):
    # Turn vector into composition
    a = (x / x.sum()).replace(0., 10**-20)
    return np.log(a[1:].values/a.values[0])


# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_anchor.html#matplotlib.axes.Axes.set_anchor
# I couldn't get fig.add_axes to work with add_anchor

def add_axes(ax, x, y, w, h, anchor='C', **kwargs):
    """add axes at x,y on graph (centered) with width, height w,h in figure coordinates.
    Anchor can be N, S, E, W, C.
    other kwargs go to figure.add_axes"""
    # get left, bottom: convert axes coordinates to figure coordinates, shift half the w, h left
    lb = ax.figure.transFigure.inverted().transform(ax.transData.transform([x, y]))
    d = 0.5*np.array([w, h])  # offset vector: in proportion of width
    lb = lb - d + d * np.array([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]])['CNSEW'.index(anchor)]
    # ... offset vectors for N,S,E,W
    return ax.figure.add_axes([*tuple(lb), w, h], **kwargs)

    
def category_bars(ax, df, **kwargs):
    """1st column of df is plotted as outline of a bar,  intended to be the mean.
    2nd column is solid bar of a bar.
    
kwargs:
    show_xaxis: whether to shwo labels, x axis
"""
    legend = False
    show_xaxis = kwargs.pop('show_xaxis', False)
    grph = dict(kind='bar', width=0.9, ax=ax,  legend=legend)  # for all bars
    barparm = [dict(fill=False, edgecolor='k', linewidth=0.5)  # for each series in turn
              , dict(color='r', alpha=0.4)]
    for c, bp in zip(df.columns, barparm):
        df[[c]].plot(**grph, **bp)
    
    if show_xaxis:        # nice x tick labels if they fit: rotate into the bar, and padd them
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, ha='left', rotation_mode="anchor")  # rotate up into bar
        ax.set_xticklabels([' '*4+t.get_text() for t in ax.get_xticklabels()])  # add padd to put start up above x axis
    else:
        ax.get_xaxis().set_visible(False)
    # % fmt on x axis
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:,.0%}'.format))
    return ax


class Decomposer(object):
    """Class with decomposition analysis and presentation functions"""
    
    def __init__(self, df, coda=False):
        """df is a data frame with years across columns, item by which split down the rows.
kwargs:

coda=False: whether to transform into a composition before decomposition

"""
        a = df.div(df.sum()) # the distribution
        if coda:
            a = a.apply(get_comp) # applies to each column
        mu = a.mean(axis=1)  # the mean by year (across columns)
        ac = a.sub(mu, axis=0) # center: subtract the mean (ac = a centered)
        # svd:
        u, s, v = np.linalg.svd(ac.values, full_matrices=False)
        v = v.T
        
        # row labels to use
        rws = df.index
        if coda: # insert 0 value at start for the default category
            u  = np.insert(u, 0, 0., axis=0) 
            mu = np.insert(mu.values, 0, 0.) # is vector, don't need axis
        
        # Save the main pieces here from the set-up
        
        self.df = df
        self.u  = pd.DataFrame(u, index = rws)  # the compositions: with zero value at start
        self.v  = pd.DataFrame(v, index = a.columns)
        self.s  = pd.Series(s)
        self.sv = pd.DataFrame(v.dot(np.diag(s)), index = a.columns) # years' cooordinates: putting singular values all here, no sqrt
        self.mu = pd.Series(mu, index=rws)
        self.a  = a
        self.ac = ac

        
    def plot2d(self, show_xaxis=False, ax_label_scale=1.0, ax=None):
        """
        Plot decomposition in 2D, with axis labels
            args:
            show_xaxis = False: to show xaxis in the overall bar charts on the x and y axis labels
            ax_label_scale = 1.0: the labels are this scale times the singular vector for that axis
            ax=None: axes on which to draw
        """
        
        # for convenience
        u, sv, mu = self.u, self.sv, self.mu

        # Get axis on which to draw
        if ax==None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.figure
        
        _n = u.index.name # name of item by which years split out

        # Plot the line of the years (the time series)
        sv.plot(0, 1, style='.-', ax=ax, title=_n, legend=False, grid=True)

        # Label the points by year
        for i in sv.index: # year goes down rows
            ax.text(sv.loc[i,0], sv.loc[i,1], str(i))

        # Show more space on the graph to be sure the points won't be on the edges
        _c = 1.1  # to scale: center 0, allow margin around edges
        ax.set_xlim(*(_c * sv[0].map(np.abs).max() * np.array([-1, 1])))
        # keep same scale on y axes so it's clear that the x axis covers more spread
        ax.set_ylim(*(_c * sv[1].map(np.abs).max() * np.array([-1, 1])))

        # for x, y axes labels: a dataframe of the bars
        df_bars = pd.DataFrame({'mean':mu
                                , 'x': mu + ax_label_scale * u[0] * sv[0].max() # x axis label
                                , 'y': mu + ax_label_scale * u[1] * sv[1].max()  # y axis label
                               })

        # not used at moment: the type of axis label
        k = {'issue_age':'line', 'duration':'line'}.get(_n, 'bar')
        
        # Plot x axis label graph.
        scales= [ax.get_xlim()[1], ax.get_ylim()[1]] # scales of singular vectors to use
        category_bars(add_axes(ax, ax.get_xlim()[1]*1.02, 0, 0.3, 0.3, anchor='E', alpha=0.02)
                      , df_bars[['mean', 'x']]
                      , show_xaxis=show_xaxis)

        # Plot the y axis label.
        category_bars(add_axes(ax, 0, ax.get_ylim()[1]*1.02, 0.3, 0.3, anchor='S', alpha=0.02)
                      , df_bars[['mean', 'y']]
                      , show_xaxis=show_xaxis)    

        # Axis lines through the origin
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

        return fig, ax

