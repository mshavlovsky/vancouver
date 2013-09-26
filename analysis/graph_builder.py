import numpy as np
from random import choice

class Graph:
    
    def __init__(self, items, users, reviews=5,
                 use_mlestimator_user_variance=False):
        """Initializes a random graph between the specified users
        and items, for the specified number of reviews."""
        self.items = items
        self.users = users
        self.n_reviews = reviews
        self.n_items = len(items)
        self.n_users = len(users)
        
        # Builds a random graph.
        self.under_allocated_users = []
        self.under_allocated_items = []
        
        for i in range(self.n_items * self.n_reviews):
            item = self.pick_item()
            user = self.pick_user()
            user.add_item(item)
            item.add_user(user)
        self.use_mle = use_mlestimator_user_variance
        

    def pick_user(self):
        if self.under_allocated_users == []:
            self.under_allocated_users = [u for u in self.users]
        i = choice(range(len(self.under_allocated_users)))
        return self.under_allocated_users.pop(i)
            
    def pick_item(self):
        if self.under_allocated_items == []:
            self.under_allocated_items = [it for it in self.items]
        i = choice(range(len(self.under_allocated_items)))
        return self.under_allocated_items.pop(i)
            
        
