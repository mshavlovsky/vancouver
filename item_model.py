import numpy as np
import numpy.random as npr


class Item:
    def __init__(self, stdev=1.0):
        self._stdev = stdev
        self.q = npr.normal(loc=0.0, scale=stdev)
        self.users = []
        
    def add_user(self, user):
        self.users.append(user)
        
