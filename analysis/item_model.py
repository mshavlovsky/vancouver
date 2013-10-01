import numpy as np
import numpy.random as npr


class Item:
    def __init__(self, stdev=1.0, frac=0, difficulty=0.0, true_quality=None):
        self._stdev = stdev
        if npr.uniform() < frac:
            self.difficulty = difficulty
        else:
            self.difficulty = 0.0
        if true_quality is None:
            self.true_quality = npr.normal(scale=stdev)
        else:
            self.true_quality = true_quality
        self.users = []
        self.grade = None

    @property
    def q(self):
        if self.difficulty <= 0:
            return self.true_quality
        else:
            return self.true_quality + npr.normal(scale=self.difficulty)

    def add_user(self, user):
        self.users.append(user)
