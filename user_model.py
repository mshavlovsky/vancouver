import numpy as np
import numpy.random as npr


class User:
    
    def __init__(self, bias_stdev=0.2, eval_stdev=0.2, bimodal=False, frac=0.1):
        """Initializes the precision and bias of a user.  Useful only for simulation."""
        # Chooses the bias of the user
        self.true_bias = npr.normal(scale=bias_stdev)
        # Chooses the variance of the user.
        if bimodal:
            # 10% of the students are responsible for 90% of the trouble,
            # where 10% is the fraction.
            # This code keeps the standard deviation as specified, but explains
            # it via a bimodal distribution, with values s and s / frac.
            s = eval_stdev * frac / (1.0 + frac - frac * frac)
            if npr.uniform() < frac:
                self.prec = s / frac
            else:
                self.prec = s                
        else:
            self.prec = npr.pareto(2.0) * eval_stdev
        # List of items it judged.
        self.items = []
        # Dictionary mapping each item, to the grade assigned by the user.
        self.grade = {}
        
    def judge(self, item):
        """Judges an item, according to the random model of the user.
        Useful in simulations only, obviously."""
        return item.q + self.true_bias + npr.normal(scale=self.prec)
    
    def add_item(self, item):
        """Adds an item, judging it.  Useful only in simulation mode,
        obviously."""
        self.items.append(item)
        self.grade[item] = self.judge(item)
    