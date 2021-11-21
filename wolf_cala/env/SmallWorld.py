import gym
import numpy as np
import networkx as nx
import copy


class SmallWorld(gym.Env):
    def __init__(self,num_media,k,p):
        self.action_space = gym.spaces.Discrete(1)
        self.num_gossiper = 200 #200   #小世界网络中的结点个数
        self.num_media = num_media#2
        self.k = k #10 #小世界网络中边的个数   10   199
        self.p =p # 0.2 #0.2   #0.2



    def reset(self):
        #创建小世界网络
        self.sw = nx.watts_strogatz_graph(self.num_gossiper,self.k,self.p)
        #将网络转成邻接矩阵
        self.swm = nx.to_numpy_matrix(self.sw)
        #初始化Gossiper和Media的观点
        #按照均匀分布抽样
        #self.opinion_gossiper = np.random.rand(self.num_gossiper)
        # 测试时的观点
        self.opinion_gossiper = np.genfromtxt("dataset/gossiper_opinions.csv", delimiter=',', usecols=(1),
                                              skip_header=1)

    def reset_true(self):
        path = 'dataset/facebook_combined.txt'
        data = open(path, 'r')
        node_x = []
        node_y = []
        for line in data:
            ln = line.split(' ')
            ln[1] = ln[1].replace('\n', '').strip()
            node_x.append(int(ln[0]))
            node_y.append(int(ln[1]))
        max_node_x = max(node_x)
        max_node_y = max(node_y)
        self.num_node = np.where(np.greater(max_node_x, max_node_y), max_node_x, max_node_y) + 1
        self.num_neigh = [0 for i in range(self.num_node)]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(node_x)
        self.graph.add_nodes_from(node_y)
        for i in range(len(node_x)):
            self.graph.add_edge(node_x[i], node_y[i])
        graph_martix = nx.to_numpy_matrix(self.graph)
        self.graph_array = np.array(graph_martix)

        #self.opinion_gossiper = np.random.rand(self.num_node)  #取值范围[0,1)的均匀分布
        #测试时的观点
        self.opinion_gossiper = np.genfromtxt("dataset/gossiper_opinions_facebook.csv", delimiter=',', usecols=(1), skip_header=1)
        for i in range(self.num_node):
            if i in node_x:
                self.num_neigh[i] = node_x.count(i)



    def step(self, action):
        pass

def run_env():
    env = SmallWorld()
    #env.reset()
    env.reset_true()


if __name__ == '__main__':
    run_env()