import numpy as np
from random import choice

class Graph:
    
    def __init__(self, items, users, reviews=5, compute_item_variance=False,
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
            user = self.pick_user(item)
            user.add_item(item)
            item.add_user(user)
        self.compute_item_variance = compute_item_variance
        self.use_mle = use_mlestimator_user_variance


    @classmethod
    def get_full_graph(cls, items, users):
        graph = cls([], [])
        graph.items = items
        graph.users = users
        graph.n_items = len(items)
        graph.n_users = len(users)
        for j in xrange(len(users)):
            user = users[j]
            for i in xrange(len(items)):
                item = items[i]
                if j == len(users) - 1 and i == len(items) - 1:
                    continue
                user.add_item(item)
                item.add_user(user)
        return graph


    def pick_user(self, item):
        counter = 0
        while True:
            if self.under_allocated_users == []:
                self.under_allocated_users = [u for u in self.users]
            i = choice(range(len(self.under_allocated_users)))
            if item not in self.under_allocated_users[i].items:
                return self.under_allocated_users.pop(i)
            counter += 1
            if counter > 100000 and counter < 200000:
                self.under_allocated_users = []
            if counter >= 200000:
                raise Exception("Remember, a user cannot grade an item twice!")


    def pick_item(self):
        while True:
            if self.under_allocated_items == []:
                self.under_allocated_items = [it for it in self.items]
            i = choice(range(len(self.under_allocated_items)))
            # We don't want to assign an item two times to the same user
            if len(self.under_allocated_items[i].users) <= len(self.users):
                return self.under_allocated_items.pop(i)
            else:
                self.under_allocated_items = []


    def __repr__(self):
        s = "Graph \n"
        for u in self.users:
            for it in u.items:
                s = s + "user %s -> item %s, grade %s\n" % (str(u)[-9:-1], str(it)[-9: -1], u.grade[it])
        return s

