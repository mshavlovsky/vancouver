import numpy as np
import numpy.random as npr


class User:
    
    def __init__(self, bias_stdev=0.2, eval_stdev=0.2):
        
        self.bias = npr.normal(scale=bias_stdev)
        self.prec = eval_stdev
        self.items = []
        self.grade = {}
        
    def judge(self, item):
        return item.q + self.bias + npr.normal(scale=self.prec)
    
    def add_item(self, item):
        self.items.append(item)
        self.grade[item] = self.judge(item)
    