import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def get_tree_bins(X, y, obs, leaves):
    
    # список пороговых значений для разделения деревом
    
    tree_model = DecisionTreeClassifier(max_leaf_nodes=leaves, 
                                        min_samples_leaf=int(obs/20 +1), 
                                        random_state=2)
    tree_model.fit(X, y)
    
    return [-np.inf] + list(np.sort(tree_model.tree_.threshold[tree_model.tree_.feature==0])) + [np.inf]

def get_monotonic_bins(var, df, target_name, goods, bads):
    
    df_var = df.loc[df[var].notnull(), [target_name, var]]
    X = df_var[var].values.reshape(-1, 1)
    y = df_var[target_name]
    obs = df[var].shape[0]
    
    for leaf in [7, 6, 5, 4, 3, 2]:
        try:
            bins = get_tree_bins(X, y, obs, leaf)
            woes = df_var.groupby(pd.cut(df_var[var], bins), observed=False)[target_name].agg(woe, goods=goods, bads=bads)
            
            if np.sign(woes.diff()).nunique() == 1:
                return bins
        except:
            pass        # если не получается разбить на бины (текст напрмер), он их пропустит
        
def woe(x, goods, bads):
    
    #n_goods = goods
    #n_bads = bads
    
    if (1-x).sum() == 0: # all bads
        return np.log((((1-x).sum() + 1) / goods) / (x.sum() / bads))
    elif x.sum() > 0:
        return np.log(((1-x).sum() / goods) / (x.sum() / bads))
    else:
        return np.log((((1-x).sum()) / goods) / ((x.sum() + 1) / bads))
    

def get_woe_bins(var, df, target_name, BINS, goods, bads):
    
    try:
        woes = df[[target_name, var]].groupby(pd.cut(df[var], BINS[var]) \
                                                .cat.add_categories('NULL') \
                                                .fillna('NULL'), observed=False) \
                                                [target_name].agg(woe, goods=goods, bads=bads)
        return woes
    
    except TypeError as exp:
        print(var)
        raise exp

def transform2woe(var, df, BINS_WOE):
    transformed = pd.cut(df[var], BINS_WOE[var]['BINS']).cat.add_categories('NULL') \
                                                        .fillna('NULL') \
                                                        .map(BINS_WOE[var]['WOES'])
    return transformed
        
    