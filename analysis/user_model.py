import numpy as np
import numpy.random as npr


class User:
    
    def __init__(self, bias_stdev=0.2, eval_stdev=0.2, mode='pareto', frac=0.1, 
                 pareto_shape=1.4, gamma_shape=3):
        """Initializes the precision and bias of a user.  Useful only for simulation."""
        # Chooses the bias of the user
        self.true_bias = 0
        if bias_stdev > 0:
            self.true_bias = npr.normal(scale=bias_stdev)
        # Chooses the variance of the user.
        if mode == 'bimodal':
            # 10% of the students are responsible for 90% of the trouble,
            # where 10% is the fraction.
            # This code keeps the standard deviation as specified, but explains
            # it via a bimodal distribution, with values s and s / frac.
            s = eval_stdev * eval_stdev * frac / (1.0 + frac - frac * frac)
            if npr.uniform() < frac:
                self.prec = (s / (frac * frac)) ** 0.5
            else:
                self.prec = s ** 0.5
        elif mode == 'pareto':
            # The mean of a pareto distribution of shape a is 1 / (a - 1)
            # Here, we use the pareto distribution to sample the variance.
            
            prec_sq = npr.pareto(pareto_shape) * eval_stdev * eval_stdev * (pareto_shape - 1.0)
            self.prec = prec_sq ** 0.5
        else:
            # Gamma.
            prec_sq = npr.gamma(gamma_shape, scale=eval_stdev)
            self.prec = prec_sq * prec_sq
            
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
    
