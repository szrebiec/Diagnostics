#######################################################################################################################
# Base Diagnostics functions
#
#
# Step 0: Header
#      1: LorenzCurve
#######################################################################################################################

##### step 0: Header #####

import pandas as pd
import numpy as np

##### Step 1: Lorenz Curve #####

def LorenzCurve(df,
                title = 'Lorenz Curve',
                ascending = True, #arg for backwards Lorenz Curve
                LVarName = 'future_incurred',
                WeightName = 'one',
                ScoreName = 'Pred',
                xlabel = 'CDF Weight',
                ylabel = 'CDF loss'):

    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt

    Local = df[[LVarName, WeightName, ScoreName]]
    Local.sort_values(ScoreName, ascending = ascending, inplace = True)
    Local['Baseline'] = Local[WeightName].cumsum() / Local[WeightName].sum()
    Local['Model Performance'] = Local[LVarName].cumsum() / Local[LVarName].sum()
    fig, ax = plt.subplots(figsize=(12, 10))
    ModelGini = 2* abs(metrics.auc(Local['Baseline'], Local['Model Performance']) -
                       metrics.auc(Local['Baseline'], Local['Baseline']) )
    ax.plot(Local['Baseline'],  Local['Model Performance'],
            label='Model performance, gini = {ModelGini}'.format(ModelGini = ModelGini),
            linestyle='-',)
    ax.plot(Local['Baseline'],  Local['Baseline'],
            label='Baseline performance,',
            linestyle='-',)

    ax.legend(loc=4)
    ax.set_xlim(0,1)
    ax.set_ylim(0 , 1)
    ax.set_yticks(np.arange(0, 1.1,.1))
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


