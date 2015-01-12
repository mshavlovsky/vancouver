#!/usr/bin/python

import numpy as np
import unittest


# Do we debias grades?
DEBIAS = True
# Aggregation using median?
AGGREGATE_BY_MEDIAN = False
# Basic precision, as multiple of standard deviation.
BASIC_PRECISION = 0.00001


class User:
    
    def __init__(self, name):
        """Initializes a user."""
        self.name = name
        self.items = []
        self.grade = {}
        # These are fake, used only for debugging.
        self.prec = 1.0
        self.true_bias = 0.0
        
    def add_item(self, it, grade):
        self.items.append(it)
        self.grade[it] = grade
        

class Item:
    
    def __init__(self, id):
        self.id = id
        self.users = []
        self.grade = None
        # This is fake, used only for debugging.
        self.q = 0.0

    def add_user(self, u):
        self.users.append(u)
        

class Graph:
    
    def __init__(self):
        
        self.items = []
        self.users = []
        self.user_dict = {}
        self.item_dict = {}
        
    def add_review(self, username, item_id, grade):
        # Gets, or creates, the user. 
        if username in self.user_dict:
            u = self.user_dict[username]
        else:
            u = User(username)
            self.user_dict[username] = u
            self.users.append(u)
        # Gets, or creates, the item.
        if item_id in self.item_dict:
            it = self.item_dict[item_id]
        else:
            it = Item(item_id)
            self.item_dict[item_id] = it
            self.items.append(it)
        # Adds the connection between the two.
        u.add_item(it, grade)
        it.add_user(u)
        
    def get_user(self, username):
        return self.user_dict.get(username)
    
    def get_item(self, item_id):
        return self.item_dict.get(item_id)
                

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
    if AGGREGATE_BY_MEDIAN:
        if weights is not None:
            return median_aggregate(v, weights=weights)
        else:
            return median_aggregate(v)
    else:
        if weights is not None:
            return np.average(v, weights=weights)
        else:
            return np.average(v)


def median_aggregate(values, weights=None):
    if len(values) == 1:
        return values[0]
    if weights is None:
        weights = np.ones(len(values))
    # Sorts. 
    vv = []
    for i in range(len(values)):
        if weights[i] > 0:
            vv.append((values[i], weights[i]))
    if len(vv) == 0:
        return values[0]
    if len(vv) == 1:
        x, _ = vv[0]
        return x
    vv.sort()
    v = np.array([x for x, _ in vv])
    w = np.array([y for _, y in vv])
    # print 'v', v, 'w', w
    # At this point, the values are sorted, they all have non-zero weight,
    # and there are at least two values.
    half = np.sum(w) / 2.0
    below = 0.0
    i = 0
    while i < len(v) and below + w[i] < half:
        below += w[i]
        i += 1
    # print 'i', i, 'half', half, 'below', below
    if half < below + 0.5 * w[i]:
        # print 'below'
        if i == 0:
            return v[0]
        else:
            alpha = half - below
            beta = below + 0.5 * w[i] - half
            # print 'alpha', alpha, 'beta', beta
            return (beta * (v[i] + v[i - 1]) / 2.0 + alpha * v[i]) / (alpha + beta)
    else:
        # print 'above'
        if i == len(v) - 1:
            # print 'last'
            return v[i]
        else:
            alpha = half - below - 0.5 * w[i]
            beta = below + w[i] - half
            # print 'alpha', alpha, 'beta', beta
            return (beta * v[i] + alpha * (v[i] + v[i + 1]) / 2.0) / (alpha + beta)


class Msg():
    def __init__(self):
        pass


def _propagate_from_items(graph):
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
            variances = []
            for m in it.msgs:
                if m.user != u:
                    grades.append(m.grade)
                    variances.append(m.variance)
            variances = np.array(variances)
            weights = 1.0 / (BASIC_PRECISION + variances)
            weights /= np.sum(weights)
            msg = Msg()
            msg.item = it
            msg.grade = aggregate(grades, weights=weights)
            # Now I need to estimate the variance of the grade. 
            # Estimates the standard deviation of the user, from the
            # other judged items.
            msg.variance = 1.0 / np.sum(1.0 / (BASIC_PRECISION + variances))
            # The message is ready for enqueuing.
            u.msgs.append(msg)


def _propagate_from_users(graph):
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
                if u.is_instructor:
                    u.bias = 0.0
                else:
                    for m in u.msgs:
                        if m.item != it:
                            weights.append(1 / (BASIC_PRECISION + m.variance))
                            given_grade = u.grade[m.item]
                            other_grade = m.grade
                            biases.append(given_grade - other_grade)
                    u.bias = aggregate(biases, weights=weights)
            else:
                u.bias = 0.0
            # The grade is the grade given, de-biased.
            msg.grade = u.grade[it] - u.bias
            # Estimates the standard deviation of the user, from the
            # other judged items.
            if u.is_instructor:
                msg.variance = 0.000001
            else:
                variance_estimates = []
                weights = []
                for m in u.msgs:
                    if m.item != it:
                        it_grade = u.grade[m.item] - u.bias
                        variance_estimates.append((it_grade - m.grade) ** 2.0)
                        weights.append(1.0 / (BASIC_PRECISION + m.variance))
                msg.variance = aggregate(variance_estimates, weights=weights)
            # The message is ready for enqueuing.
            it.msgs.append(msg)
                

def _aggregate_item_messages(graph):
    """Aggregates the information on an item, computing the grade
    and the variance of the grade."""
    item_values = {}
    all_weights = [] # debug
    weight_vs_error = []
    stdev_vs_true_stdev = []
    stdev_vs_error = []
    for it in graph.items:
        grades = []
        variances = []
        for m in it.msgs:
            grades.append(m.grade)
            variances.append(m.variance)
        variances = np.array(variances)
        weights = 1.0 / (BASIC_PRECISION + variances)
        weights /= np.sum(weights)
        all_weights.append(weights)
        it.grade = aggregate(grades, weights=weights)
        it.variance = 1.0 / np.sum(1.0 / (BASIC_PRECISION + variances))
        item_values[it] = it.grade
        # Debug
        for i, m in enumerate(it.msgs):
            stdev_vs_error.append((variances[i] ** 0.5, abs(m.grade - it.grade)))
            stdev_vs_true_stdev.append((variances[i] ** 0.5, m.user.prec))
            weight_vs_error.append((weights[i], abs(it.q - m.grade)))
    return item_values, all_weights, weight_vs_error, stdev_vs_error, stdev_vs_true_stdev


def _aggregate_user_messages(graph):
    """Aggregates the information on a user, computing the 
    variance and bias of a user."""
    for u in graph.users:
        biases = []
        weights = []
        # Estimates the bias.
        if DEBIAS:
            if u.is_instructor:
                u.bias = 0.0
            else:
                for m in u.msgs:
                    weights.append(1 / (BASIC_PRECISION + m.variance))
                    given_grade = u.grade[m.item]
                    other_grade = m.grade
                    biases.append(given_grade - other_grade)
                u.bias = aggregate(biases, weights=weights)
        else:
            u.bias = 0.0
        # Estimates the grade for each item.
        if u.is_instructor:
            u.variance = 0.000001
        else:
            variance_estimates = []
            weights = []
            for m in u.msgs:
                it_grade = u.grade[m.item] - u.bias
                variance_estimates.append((it_grade - m.item.grade) ** 2.0)
                weights.append(1.0 / (BASIC_PRECISION + m.variance))
            u.variance = aggregate(variance_estimates, weights=weights)

    
def evaluate_items(graph, n_iterations=20, do_plots=False):
    """Evaluates items using the reputation system iterations."""
    # Builds the initial messages from users to items. 
    for it in graph.items:
        it.msgs = []
        for u in it.users:
            m = Msg()
            m.user = u
            m.grade = u.grade[it]
            if u.is_instructor:
                m.variance = 0.000001
            else:
                m.variance = 1.0
            it.msgs.append(m)
    # Does the propagation iterations.
    for i in range(n_iterations):
        _propagate_from_items(graph)
        _propagate_from_users(graph)

        # _aggregate_item_messages(graph)
        # _aggregate_user_messages(graph)
        # print
        # print 'iteration', i
        # print 'users'
        # for u in graph.users:
        #     is_neighbour = 'NOT instuctor\'s neighbour'
        #     if u.is_instructors_neighbour():
        #         is_neighbour = 'instuctor\'s neighbour'
        #     print u.bias, u.variance, is_neighbour
    # Does the final aggregation step.
    r, ws, w_vs_e, s_vs_e, s_vs_ts = _aggregate_item_messages(graph)
    _aggregate_user_messages(graph)
    for u in graph.users:
        print 'est. bias', u.bias, 'true bias', u.true_bias
    if do_plots:
        plot_graph(graph, ws, w_vs_e, s_vs_e, s_vs_ts)
    return r


def plot_graph(graph, ws, w_vs_e, s_vs_e, s_vs_ts):
    from matplotlib import pyplot as plt
    def unzip(l):
        return [x for x, _ in l], [x for _, x in l]
    # Plots user variance, estimated vs. true.
    plt.subplot(2,4,1)
    var_plot = []
    for u in graph.users:
        var_plot.append((u.prec, u.variance ** 0.5))
    var_plot.sort()
    x, y = unzip(var_plot)
    plt.plot(x, y, 'ro')
    plt.title('user stdev, est vs true')
    # Plots user bias, estimated vs. true. 
    plt.subplot(2,4,2)
    var_plot = []
    for u in graph.users:
        var_plot.append((u.true_bias, u.bias))
    var_plot.sort()
    x, y = unzip(var_plot)
    plt.plot(x, y, 'ro')
    plt.title('user bias, est vs true')
    # Plots item true value vs. estimated.
    plt.subplot(2,4,5)
    var_plot = []
    for it in graph.items:
        var_plot.append((it.q, it.grade))
    var_plot.sort()
    x, y = unzip(var_plot)
    plt.plot(x, y, 'ro')
    plt.title('item value, est vs true')
    # Plots item error vs. item variance. 
    plt.subplot(2,4,6)
    var_plot = []
    for it in graph.items:
        var_plot.append((it.variance ** 0.5, abs(it.grade - it.q)))
    var_plot.sort()
    x, y = unzip(var_plot)
    plt.plot(x, y, 'ro')
    plt.title('item error, est vs true')
    # Plots the stdev vs weight distribution.
    plt.subplot(2,4,3)
    y, x = unzip(s_vs_e)
    plt.plot(x, y, 'ro')
    # plt.xscale('log')
    plt.title('stdev vs. error')
    # Plots the stdev vs true stdev distribution.
    plt.subplot(2,4,4)
    y, x = unzip(s_vs_ts)
    plt.plot(x, y, 'ro')
    # plt.xscale('log')
    plt.title('stdev vs. true stdev')
    # Plots weight vs. error.
    plt.subplot(2,4,7)
    y, x = unzip(w_vs_e)
    plt.plot(x, y, 'ro')
    # plt.xscale('log')
    plt.title('weight vs. error')
    plt.show()


class TestMedian(unittest.TestCase):
    
    def test_median_0(self):
        values = [1.0, 3.0, 2.0]
        weights = [1.0, 1.0, 1.0]
        m = median_aggregate(values, weights=weights)
        self.assertAlmostEqual(m, 2.0, 4)

    def test_median_1(self):
        values = [1.0, 3.0, 2.0]
        weights = [1.0, 1.0, 2.0]
        m = median_aggregate(values, weights=weights)
        self.assertAlmostEqual(m, 2.0, 4)

    def test_median_2(self):
        values = [1.0, 3.0, 2.0]
        weights = [1.0, 2.0, 1.0]
        m = median_aggregate(values, weights=weights)
        self.assertAlmostEqual(m, 2.5, 4)

    def test_median_3(self):
        values = [1.0, 3.0, 2.0]
        weights = [1.0, 2.0, 2.0]
        m = median_aggregate(values, weights=weights)
        self.assertAlmostEqual(m, 2.25, 4)


class test_reputation(unittest.TestCase):
    
    def test_rep_1(self):
        g = Graph()
        g.add_review('luca', 'pizza', 8.0)
        g.add_review('luca', 'pasta', 9.0)
        g.add_review('luca', 'pollo', 5.0)
        g.add_review('mike', 'pizza', 7.5)
        g.add_review('mike', 'pollo', 8.0)
        g.add_review('hugo', 'pizza', 6.0)
        g.add_review('hugo', 'pasta', 7.0)
        g.add_review('hugo', 'pollo', 7.5)
        g.add_review('anna', 'pizza', 7.0)
        g.add_review('anna', 'pasta', 8.5)
        g.add_review('anna', 'pollo', 5.5)
        evaluate_items(g)
        print 'pasta', g.get_item('pasta').grade
        print 'pizza', g.get_item('pizza').grade
        print 'pollo', g.get_item('pollo').grade
        print 'luca', g.get_user('luca').variance
        print 'mike', g.get_user('mike').variance
        print 'hugo', g.get_user('hugo').variance
        print 'anna', g.get_user('anna').variance

    def test_rep_2(self):
        import user_model
        import item_model
        import graph_builder
        N_USERS = 50
        N_ITEMS = 50
        N_REVIEWS = 6
        users = [user_model.User(true_bias=2, true_precision=1.0)
                 for u in range(N_USERS)]
        for i in [0]:
            users[i].true_bias = 0.0
            users[i].prec = 0.000001
            users[i].is_instructor = True
        items = [item_model.Item(stdev=5.0) for i in range(N_ITEMS)]
        graph = graph_builder.Graph(items, users, reviews=N_REVIEWS)
        evaluate_items(graph, n_iterations=50)

if __name__ == '__main__':
    unittest.main()
