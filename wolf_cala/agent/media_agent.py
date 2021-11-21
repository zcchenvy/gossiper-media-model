import numpy as np
from agent.config import config
from agent.update_strategy import Update_strategy

class Media_Agent(object):
    def __init__(self, u_ini, sigma_ini):
        self.d_m = config.d_m
        self.d_g = config.d_g
        self.alpha_sigma = config.alpha_sigma
        self.alpha_ub = config.alpha_ub
        self.alpha_us = config.alpha_us
        #self.deta = deta

        self.Q = 0
        self.y_mean = u_ini
        self.c = 0
        self.sigma_ini = sigma_ini
        self.u = u_ini
        self.sigma = sigma_ini

    def choose_action(self):
        media_opinion = np.random.normal(self.u, self.sigma)
        media_opinion = max(min(media_opinion, 1), 0)
        return media_opinion

    def opinion(self):
        return self.y_mean

    def update(self, media_opinion, reward):
        #更新u
        self.c += 1
        if reward > self.Q:
            self.u = self.u + self.alpha_ub * (media_opinion - self.u)
        else:
            self.u = self.u + self.alpha_us * (media_opinion - self.u)
        # 更新sigma
        self.sigma = max(self.sigma + self.alpha_sigma * (reward - self.Q) * (abs(media_opinion - self.u) - self.sigma)
                         - 0.0005,  0.01)
        # 更新Q
        if reward > self.Q:
            self.Q = self.Q + self.alpha_ub * (reward - self.Q)
        else:
            self.Q = self.Q + self.alpha_us * (reward - self.Q)
        # 更新y_mean
        self.y_mean = ((self.c - 1) / self.c) * self.y_mean + (1 / self.c) * media_opinion


    def reset(self):
        self.sigma = self.sigma_ini
        self.y_mean = 0
        self.c = 0


