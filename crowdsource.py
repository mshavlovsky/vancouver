#!/usr/bin/python

import average_voting
import user_model
import item_model
import graph_builder
import reputation
import numpy as np

N_USERS = 50
N_ITEMS = 50
N_REVIEWS = 10
BIAS_STDEV = 0.001
EVAL_STDEV = 0.2
FRACTION_BAD = 0.1

def eval_quality(values):
    diffs = [values[it] - it.q for it in items]
    return np.std(diffs)

# Builds a graph between users and items.

avs = []
rvs = []
for i in range(1):
    users = [user_model.User(bias_stdev=BIAS_STDEV, eval_stdev=EVAL_STDEV, bimodal=False, frac=FRACTION_BAD)
             for u in range(N_USERS)]
    items = [item_model.Item() for i in range(N_ITEMS)]
    graph = graph_builder.Graph(items, users, reviews=N_REVIEWS)
    # Evaluates this according to simple average. 
    values_via_avg = average_voting.evaluate_items(graph)
    av = eval_quality(values_via_avg)
    print "Via average: ", av
    avs.append(av)
    # Evaluates this according to the reputation system.
    values_via_rep = reputation.evaluate_items(graph, do_plots=True)
    rv = eval_quality(values_via_rep)
    print "Via reputation:", rv
    rvs.append(rv)
print "Mean via average:", np.average(avs)
print "Mean via reputation:", np.average(rvs)



