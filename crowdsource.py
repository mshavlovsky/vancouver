#!/usr/bin/python

import average_voting
import user_model
import item_model
import graph_builder
import reputation
import numpy as np

N_USERS = 50
N_ITEMS = 50
BIAS_STDEV = 0.2
EVAL_STDEV = 0.2

def eval_quality(values):
    diffs = [values[it] - it.q for it in items]
    return np.std(diffs)

# Builds a graph between users and items.
users = [user_model.User(bias_stdev=BIAS_STDEV, eval_stdev=BIAS_STDEV)
         for u in range(N_USERS)]
items = [item_model.Item() for i in range(N_ITEMS)]
graph = graph_builder.Graph(items, users)

# Evaluates this according to simple average. 
values_via_avg = average_voting.evaluate_items(graph)
print "Via average: ", eval_quality(values_via_avg)
# Evaluates this according to the reputation system.
values_via_rep = reputation.evaluate_items(graph)
print "Via reputation:", eval_quality(values_via_rep)



