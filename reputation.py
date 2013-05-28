
import numpy as np

# Do we debias grades?
DEBIAS = True

# Basic precision, as multiple of standard deviation.
BASIC_PRECISION = 0.1

# Evaluates each item via average voting.
def avg_evaluate_items(graph):
    item_value = {}
    for it in graph.items:
        grades = []
        for u in it.users:
            grades.append(u.grade[it])
        item_value[it] = aggregate(grades)
    return item_value


def aggregate(v, weights=None):
    """Aggregates using either average or median."""
    # TODO(luca): aggregate using medians as well.
    if weights is not None:
        return np.average(v, weights=weights)
    else:
        return np.average(v)


class Msg():
    def __init__(self):
        pass


def propagate_from_items(graph):
    """Propagates the information from items to users."""
    # First, clears all incoming messages.
    for u in graph.users:
        u.msgs = []
    # For each item, gives feedback to the users.
    for it in graph.items:
        # For each user that evaluated the item, reports to that user the following
        # quantities, computed from other users:
        # Average/median
        # Standard deviation
        # Total weight
        for u in it.users:
            grades = []
            weights = []
            variances = []
            for m in it.msgs:
                if m.user != u:
                    grades.append(m.grade)
                    variances.append(m.stdev ** 2.0)
            variances = np.array(variances)
            weights = 1.0 / (BASIC_PRECISION + variances)
            weights /= np.sum(weights)
            msg = Msg()
            msg.item = it
            msg.grade = aggregate(grades, weights=weights)
            # Now I need to estimate the standard deviation of the grade. 
            variance = np.sum(variances * weights * weights)
            msg.stdev = variance ** 0.5
            u.msgs.append(msg)


def propagate_from_users(graph):
    """Propagates the information from users to items."""
    # First, clears the messages received in the items.
    for it in graph.items:
        it.msgs = []
    # Sends information from users to items.  
    # The information to be sent is a grade, and an estimated standard deviation.
    for u in graph.users:
        for it in u.items:
            # The user looks at the messages from other items, and computes
            # what has been the bias of its evaluation. 
            msg = Msg()
            msg.user = u
            biases = []
            weights = []
            if DEBIAS:
                for m in u.msgs:
                    if m.item != it:
                        weights.append(1 / (BASIC_PRECISION + m.stdev ** 2))
                        given_grade = u.grade[m.item]
                        other_grade = m.grade
                        biases.append(given_grade - other_grade)
                bias = aggregate(biases, weights=weights)
            else:
                bias = 0.0
            # The grade is the grade given, de-biased. 
            msg.grade = u.grade[it] - bias
            # Estimates the standard deviation of the user, from the
            # other judged items.
            stdev_estimates = []
            weights = []
            for m in u.msgs:
                if m.item != it:
                    stdev_estimates.append((msg.grade - m.grade) ** 2.0)
                    weights.append(1.0 / (BASIC_PRECISION + m.stdev ** 2.0))
            stdev_estimate = aggregate(stdev_estimates, weights=weights) ** 0.5
            msg.stdev = stdev_estimate
            msg.weight = np.sum(weights)
            # The message is ready for enqueuing.
            it.msgs.append(msg)
                

def aggregate_item_messages(graph):
    item_values = {}
    for it in graph.items:
        grades = []
        weights = []
        for m in it.msgs:
                grades.append(m.grade)
                weights.append(m.weight)
        item_values[it] = aggregate(grades, weights=weights)
    return item_values

    
def evaluate_items(graph, n_iterations=10):
    """Evaluates items using the reputation system iterations."""
    # Builds the initial messages from users to items. 
    for it in graph.items:
        it.msgs = []
        for u in it.users:
            m = Msg()
            m.user = u
            m.grade = u.grade[it]
            m.stdev = 1.0
            it.msgs.append(m)
    # Does the propagation iterations.
    for i in range(n_iterations):
        propagate_from_items(graph)
        propagate_from_users(graph)
    # Does the final aggregation step.
    return aggregate_item_messages(graph)

        