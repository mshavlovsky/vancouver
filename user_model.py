import numpy as np
import numpy.random as npr


class User:
    
    def __init__(self, bias_stdev=0.2, eval_stdev=0.2, bimodal=False):
        
        # Chooses the bias of the user
        self.bias = npr.normal(scale=bias_stdev)
        # Chooses the variance of the user.
        self.prec = npr.pareto(2.0) * eval_stdev
        self.items = []
        self.grade = {}
        
    def judge(self, item):
        return item.q + self.bias + npr.normal(scale=self.prec)
    
    def add_item(self, item):
        self.items.append(item)
        self.grade[item] = self.judge(item)
    