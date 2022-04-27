"""
This file is to be run separately from glmtools.
The models are saved as glmrun objects already, and these additional functions
are added onto the glmrun class.

"""

import networkx as nx 

#################################################################################################################
###  FROM HERE DOWN: The revised functions to plot correctly when have cross-combinations.
###  Note: All factors that share variables must be combined into a tensor of factors for comparison to A/E.
###  That means that all variables that are connected must be grouped together - and the degree must be <= 2 for it
###  to be shown here.
#################################################################################################################

def getSubgraphs(coefs):
    """Retrun groups of the variable sets of factors which share no indices between groups.  Pass output of getCoef."""
    g = nx.Graph()
    for c0, c1 in itertools.product(coefs, coefs):
        # Add the edge if they intersect, including edges to themselves so they'l show up as neighbors.
        if 0<len(set(c0.index.names).intersection(c1.index.names)):
            g.add_edge(c0.index.names, c1.index.names) # each will be added twice, so what
    sgs = [] # the subgraphs
    ns_left = set(g.nodes())
    while len(ns_left): # while some nodes are unassigned to a graph:
        # Assign the first one to a new subgraph.
        sgs.append(g.neighbors(ns_left.pop()))
        ns_left = ns_left.difference(sgs[-1]) # Take that subgraph's nodes out from the remaining nodes.
    return sgs

def addCoefs(coefs):
    """Add the two sets of coeffs. Their like-named multiindex columns will be matched.
    They are not exponentiated here.
    I need to start with a series with all the possibilities.
    NOTE: will miss adding default categories in some cases, always send 1st element in list with 
    series of zeros with full index to be populated.
    This full index could be obtained by using the field names from a pivot tabel of the source data."""
    res = coefs[0];
    for _c in coefs[1:]:
        res = res.add(_c, fill_value=0)
    return res

def compPlot(c):
    """Plot a dataframe of Factor and A/Table. If factors are univariate then is simple; if bivarate
    then plot 4 graphs.
    
    Utility function, which can be used alone.
    """
    deg = len(c.index.names) # degree of the plots
    
    plotoptions = dict(rot=45, ylim=(0.5, 2))
    if 1==deg:
        c.plot(style=['k.-', 'b.-'], **plotoptions)
    elif 2==deg:
        # Plot 4: factor, beside a/e, with one x, then the other.
        tmp = c.reset_index().pivot_table(index=c.index.names[0], columns=c.index.names[1], values=c.columns) 
        # ... aggfunc irrelevant, is exactly one value, so mean, min, sum, max, ... all same
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(8,6)
        fig.tight_layout()
        for _ax, _y, _leg in zip(ax[0], c.columns, [False, True]):# Factor or A/E
            tmp[_y].plot(title=_y, ax= _ax, legend=_leg, **plotoptions)
        for _ax, _y, _leg in zip(ax[1], c.columns, [False, True]):# Factor or A/E # Transposed other way                          
            tmp[_y].T.plot(title=_y,  ax=_ax, legend=_leg, **plotoptions)
        ax[0,1].legend(loc='upper left',bbox_to_anchor=(1,1))
        ax[1,1].legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.tight_layout()

