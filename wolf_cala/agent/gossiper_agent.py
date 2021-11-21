import numpy as np
from agent.config import config
from agent.update_strategy import Update_strategy
import copy

class Gossiper_Agents(object):
    def __init__(self, opinions_ini, social_network, mode='random'):
        self.opinions = opinions_ini
        self.social_network = social_network
        self.num_gossiper = self.social_network.shape[0]
        self.d_m = config.d_m
        self.d_g = config.d_g
        self.alpha_g = config.alpha_g
        self.alpha_m = config.alpha_m
        self.mode = mode
        self.delta = config.delta

        self.followed_media = [-1 for i in range(self.num_gossiper)]
        self.neighbors = []
        self.num_neighbors = []
        for i in range(self.num_gossiper):
            neighbor_i = []
            c = 0
            for j in range(self.num_gossiper):
                if i != j and self.social_network[i][j] == 1:
                    neighbor_i.append(j)
                    c += 1
            self.neighbors.append(neighbor_i)
            self.num_neighbors.append(c)

    def update_opinion(self, media_opinions=None):
        if self.mode == 'random':
            for i in range(self.num_gossiper):
                if self.num_neighbors[i] > 0:
                    choice_gossiper = np.random.randint(0, self.num_neighbors[i])
                    if abs(self.opinions[i]-self.opinions[choice_gossiper]) < self.d_g:
                        self.opinions[i] = self.opinions[i] + self.alpha_g * (self.opinions[choice_gossiper]-self.opinions[i])
        else:
            for i in range(self.num_gossiper):
                if self.num_neighbors[i] > 0:
                    opinions = []
                    for j in range(self.num_neighbors[i]):
                        if abs(self.opinions[i]-self.opinions[j]) < self.d_g:
                            opinions.append(copy.deepcopy(self.opinions[j]))
                    self.opinions[i] = self.opinions[i] + self.alpha_g * (np.mean(opinions)-self.opinions[i])
        if len(media_opinions) > 0:
            num_media = len(media_opinions)
            for i in range(self.num_gossiper):
                choice_media = np.random.randint(0, num_media)
                if abs(self.opinions[i] - media_opinions[choice_media]) < self.d_m:
                    self.opinions[i] = self.opinions[i] + self.alpha_m * (media_opinions[choice_media] - self.opinions[i])

    def update_followed_media(self, media_opinions):
        self.followed_media = [-1 for i in range(self.num_gossiper)]
        num_media = len(media_opinions)
        for i in range(self.num_gossiper):
            lamda = []
            for j in range(num_media):
                if abs(self.opinions[i] - media_opinions[j]) < self.d_m:
                    lamda.append(1 / max(self.delta, abs(self.opinions[i] - media_opinions[j])))
                else:
                    lamda.append(0)
                    #lamda.append(np.exp(-self.tau * abs(self.opinions[i] - media_opinions[j])))
            if sum(lamda) > 0:
                policy = [lamda[j]/sum(lamda) for j in range(num_media)]
                k = np.random.random()
                kk = 0#policy[0]
                m = 0
                for j in range(0, num_media):
                    if kk < k:
                        kk += policy[j]
                        m += 1
                self.followed_media[i] = m-1

    def step(self, media_opinions):
        self.update_opinion(media_opinions)
        self.update_followed_media(media_opinions)
        reward = np.zeros(len(media_opinions))
        for i in range(self.num_gossiper):
            if self.followed_media[i] >= 0:
                reward[self.followed_media[i]] += 1
        return reward,

    def opinions_gossipers(self):
        return self.opinions













