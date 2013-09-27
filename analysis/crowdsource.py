#!/usr/bin/python

import average_voting
import user_model
import item_model
import graph_builder
import reputation_instrumented
import copy
import numpy as np


N_USERS = 50
N_ITEMS = 50
N_REVIEWS = 6
BIAS_STDEV = 0.0
EVAL_STDEV = 0.4
FRACTION_BAD = 0.2
GAMMA_SHAPE = 2
ITEMS_DIFFICULTY = 1
ITEMS_FRACTION_BAD = 0.2
N_ITERATIONS = 100
DO_PLOTS = False

def eval_quality(values, items, only_good=False):
    if only_good:
        items = [it for it in items if it.difficulty == 0]
    diffs = [values[it] - it.true_quality for it in items]
    stdev = np.std(diffs)
    a = [values[it] for it in items]
    b = [it.true_quality for it in items]
    r = np.corrcoef(a, b)[1, 0]
    return stdev, r

# Builds a graph between users and items.

avs = []
acs = []
rvs = []
rcs = []
stdev_mle = []
corr_mle = []
for i in range(N_ITERATIONS):
    users = [user_model.User(bias_stdev=BIAS_STDEV, eval_stdev=EVAL_STDEV, mode='gamma',
                             gamma_shape=GAMMA_SHAPE, frac=FRACTION_BAD)
             for u in range(N_USERS)]
    items = [item_model.Item(frac=ITEMS_FRACTION_BAD, difficulty=ITEMS_DIFFICULTY) for i in range(N_ITEMS)]
    graph = graph_builder.Graph(items, users, reviews=N_REVIEWS)
    # Evaluates this according to simple average.
    values_via_avg = average_voting.evaluate_items(graph)
    av, ac = eval_quality(values_via_avg, items)
    #print "  Via average: ", av, ac
    avs.append(av)
    acs.append(ac)
    # Evaluates this according to the reputation system.
    values_via_rep = reputation_instrumented.evaluate_items(graph, do_plots=DO_PLOTS)
    rv, rc = eval_quality(values_via_rep, items)
    #print "  Via reputation:", rv, rc
    rvs.append(rv)
    rcs.append(rc)
    # Evaluates this according to the repuation systme using mle estimator for users' variance
    graph.use_mle = True
    values_via_rep_mle = reputation_instrumented.evaluate_items(graph, do_plots=DO_PLOTS)
    mle_v, mle_c = eval_quality(values_via_rep_mle, items)
    stdev_mle.append(mle_v)
    corr_mle.append(mle_c)

print "Stdev via average:", np.average(avs)
print "Stdev via reputation:", np.average(rvs)
print "Stdev vie reputation using mle:", np.average(stdev_mle)
print "Correlation via average:", np.average(acs)
print "Correlation via reputation:", np.average(rcs)
print "Correlation via reputation using mle:", np.average(corr_mle)


