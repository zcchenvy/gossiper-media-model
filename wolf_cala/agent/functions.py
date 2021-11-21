import random
import numpy as np
import copy
from agent.config import config

def produce_graph(opinions, d_g):
    n = len(opinions)
    graph = []
    for i in range(n):
        neighbors_i = []
        for j in range(n):
            if i != j and abs(opinions[i] - opinions[j]) < d_g:
                neighbors_i.append(copy.deepcopy(1))
            else:
                neighbors_i.append(copy.deepcopy(0))
        graph.append(copy.deepcopy(neighbors_i))
    return graph

def average_clustering(graph, trials=1000):
    n = len(graph)
    triangles = 0
    nodes = range(n)
    neighbors = []
    num_neighbors = []

    # calculate neighbors
    for i in range(n):
        neighbor_i = []
        c = 0
        for j in range(n):
            if i != j and graph[i][j] == 1:
                neighbor_i.append(j)
                c += 1
        neighbors.append(copy.deepcopy(neighbor_i))
        num_neighbors.append(c)

    for i in [int(random.random() * n) for i in range(trials)]:
        nbrs = neighbors[i]
        if len(nbrs) < 2:
            continue
        u, v = random.sample(nbrs, 2)
        if graph[u][v] == 1:
            triangles += 1
    return triangles / float(trials)


def opinion_clustering(opinions, d_g):
    graph = produce_graph(opinions, d_g)
    clustering = average_clustering(graph)
    return clustering

def choose_actions_medias(medias):
    opinions = []
    num_media = len(medias)
    for i in range(num_media):
        opinions.append(copy.deepcopy(medias[i].choose_action()))
    return opinions


def opinions_of_medias(medias):
    opinions = []
    num_media = len(medias)
    for i in range(num_media):
        opinions.append(copy.deepcopy(medias[i].opinion()))
    return opinions


def sample_gossiper_opinions(opinions):
    num_goss = len(opinions)
    gossiper_sample_index = random.sample(range(num_goss), int(config.gossiper_sample_num * num_goss))
    return [opinions[gossiper_sample_index[i]] for i in range(len(gossiper_sample_index))]


def calculate_reward(sample_gossiper_opinion, media_opinion):
    gossiper_num = len(sample_gossiper_opinion)
    media_num = len(media_opinion)
    followed_num = np.zeros(media_num, dtype=int)
    for i in range(gossiper_num):
        lamda = []
        for j in range(media_num):
            if abs(sample_gossiper_opinion[i] - media_opinion[j]) < config.d_m:
                lamda.append(copy.deepcopy(1 / max(config.delta, abs(sample_gossiper_opinion[i] - media_opinion[j]))))
            else:
                lamda.append(copy.deepcopy(0))
                # lamda.append(np.exp(-self.tau * abs(self.opinions[i] - media_opinions[j])))
        if sum(lamda) > 0:
            policy = [lamda[j] / sum(lamda) for j in range(media_num)]
            k = np.random.random()
            kk = 0
            m = 0
            for j in range(0, media_num):
                if kk < k:
                    kk += policy[j]
                    m += 1
            followed_num[m-1] += 1

    return followed_num/gossiper_num

def new_clustering(graph, opinions, d_g, trials=1000):
    n = len(graph)
    nodes = range(n)
    neighbors = []
    num_neighbors = []
    triangles = 0
    # calculate neighbors
    for i in range(n):
        neighbor_i = []
        c = 0
        for j in range(n):
            if i != j and graph[i][j] == 1:
                neighbor_i.append(j)
                c += 1
        neighbors.append(copy.deepcopy(neighbor_i))
        num_neighbors.append(c)

    for i in [int(random.random() * n) for i in range(trials)]:
        nbrs = neighbors[i]
        if len(nbrs) < 2:
            continue
        u, v = random.sample(nbrs, 2)
        if abs(opinions[u] - opinions[v]) < d_g:
            triangles += 1
    return triangles / float(trials)
