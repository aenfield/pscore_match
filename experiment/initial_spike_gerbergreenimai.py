# just load the module to make reloads easier
#from pscore_match.pscore import PropensityScore
#from pscore_match.match import Match, whichMatched
import pscore_match.pscore
import pscore_match.match

from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import plotly # for single plot comparing matched covariates

#%matplotlib inline


def plot_before_after_distributions(title, field=None, axis_limits=[0, 0.15, 0, 40]):
    if field == None:
        field = title

    density0 = gaussian_kde(all_observations[all_observations['treatment'] == 0][field])
    density1 = gaussian_kde(all_observations[all_observations['treatment'] == 1][field])
    density0_post = gaussian_kde(data_matched[data_matched['treatment'] == 0][field])
    density1_post = gaussian_kde(data_matched[data_matched['treatment'] == 1][field])

    # plt.figure(1)
    plt.subplot(121)
    xs = np.linspace(axis_limits[0], axis_limits[1], 200)

    plt.plot(xs, density0(xs), color='black')
    plt.fill_between(xs, density1(xs), color='gray')
    plt.title('Phone: before matching')
    plt.axis(axis_limits)
    plt.xlabel(title)
    plt.ylabel('Density')
    plt.legend(['Control', 'Treatment'])

    plt.subplot(122)
    plt.plot(xs, density0_post(xs), color='black')
    plt.fill_between(xs, density1_post(xs), color='gray')
    plt.title('Phone: after matching')
    plt.axis(axis_limits)
    plt.xlabel(title)
    plt.ylabel('Density')
    plt.legend(['Control', 'Treatment'])

    plt.show()


print(pscore_match.__path__)

imai = pd.read_table('https://raw.githubusercontent.com/kellieotto/pscore_match/master/pscore_match/data/GerberGreenImai.txt', sep = '\s+')
print(imai.shape)

imai.index = range(imai.shape[0])

# interaction terms
imai['PERSONS1_VOTE961'] = (imai.PERSONS==1)*imai.VOTE961
imai['PERSONS1_NEW'] = (imai.PERSONS==1)*imai.NEW

treatment = np.array(imai.PHNC1)

cov_list = ['PERSONS', 'VOTE961', 'NEW', 'MAJORPTY', 'AGE', 'WARD', 'AGE2', 'PERSONS1_VOTE961', 'PERSONS1_NEW']
covariates = imai[cov_list]

pscore = pscore_match.pscore.PropensityScore(treatment, covariates).compute()

print(pd.Series(treatment).value_counts(dropna=False))

pairs = pscore_match.match.Match(treatment, pscore)
pairs.create(method='many-to-one', many_method='knn', k=5, replace=True)

all_observations_orig = pd.DataFrame({'pscore': pscore, 'treatment': treatment, 'voted': imai.VOTED98, 'id_maybe': imai.index})
# and then add in all of the original columns, so we do covariate comparison easily below; this duplicates a few cols but I'll leave this as is now so I don't have to change the names below
all_observations = pd.concat([all_observations_orig, imai], axis=1)
print(all_observations.shape)

print(pd.crosstab(all_observations['treatment'], all_observations['voted'], margins=True))

data_matched = pscore_match.match.whichMatched(pairs, all_observations)
print(data_matched.shape)

print(pd.crosstab(data_matched['treatment'], data_matched['voted'], margins=True))

plot_specs = [{'title':'Propensity score', 'field':'pscore', 'axis_limits':[0,0.15,0,40]},
              {'title':'MAJORPTY', 'axis_limits':[-0.5,1.5,0,5]},
              {'title':'PERSONS', 'axis_limits':[-0.5,3,0,5]}]

[plot_before_after_distributions(**spec) for spec in plot_specs]

print("Done.")
