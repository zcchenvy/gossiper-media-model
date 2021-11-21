import numpy as np
from agent.config import config
from env.SmallWorld import SmallWorld
import random
import matplotlib.pyplot as plt
from agent.gossiper_agent import Gossiper_Agents
from agent.media_agent import Media_Agent
from agent.functions import *
import pandas as pd
import copy




def main(Times, num_media, k, p):
    env = SmallWorld(num_media, k, p)
    data = 1
    if data == 1:
        env.reset()
        num_goss = env.num_gossiper
        swn = np.array(env.swm)
        np.savetxt("邻接矩阵.txt", swn, fmt="%d", delimiter="   ")
    elif data == 2:
        env.reset_true()
        num_goss = env.num_node
        swn = env.graph_array
        np.savetxt("邻接矩阵.txt", swn, fmt="%d", delimiter="   ")

    gossiper_opinion = env.opinion_gossiper[:]
    print("gossiper_opinion:",gossiper_opinion)
    gossiper_social_network = swn

    gossipers = Gossiper_Agents(gossiper_opinion, gossiper_social_network)
    # medias = [Media_Agent((i+1)/(config.num_media+1), config.ini_sigma/config.num_media) for i in range(config.num_media)]
    medias = [Media_Agent(0.5, config.ini_sigma) for i in range(config.num_media)]
    # medias = [Media_Agent((i+1)/(config.num_media+1), config.ini_sigma) for i in range(config.num_media)]

    medias_opinion_list = []  # 记录media观点，画图
    print(Path + NetworkName, ",media=", config.num_media, ",d_g=", config.d_g, "d_m=", config.d_m, "y=",
          config.Y_MEAN_TRAIN_EPISODES, ",", Times)
    gossiper_opinions_list = []
    followed_num_list = []
    medias_opinion_list = []
    sigma_list = []
    average_clusterings_list = []
    for steps in range(config.TRAIN_EPISODES):
        print("第", steps, "轮实验")
        media_opinions = opinions_of_medias(medias)
        if config.num_media > 0:
            sigma = medias[0].sigma
        for j in range(config.num_media):
            medias[j].reset()
        followed_num = gossipers.step(media_opinions)

        #coefficients = opinion_clustering(gossipers.opinions, config.d_g)
        coefficients = new_clustering(gossiper_social_network, gossipers.opinions, config.d_g)
        average_clusterings_list.append(copy.deepcopy(coefficients))

        opinions_sampled = sample_gossiper_opinions(gossipers.opinions)
        for train_round in range(config.Y_MEAN_TRAIN_EPISODES):

            media_opinions = choose_actions_medias(medias)
            rewards = calculate_reward(opinions_sampled, media_opinions)
            for j in range(config.num_media):
                medias[j].update(media_opinions[j], rewards[j])

        gossiper_opinions_list.append(copy.deepcopy(gossipers.opinions_gossipers()))
        medias_opinion_list.append(copy.deepcopy(opinions_of_medias(medias)))

        followed_num_list.append(copy.deepcopy(followed_num))

        if config.num_media > 0:
            sigma_list.append(copy.deepcopy(sigma))

    if config.num_media > 0:
        statistics = [followed_num_list] + [sigma_list] + [average_clusterings_list]
    else:
        statistics = [followed_num_list] + [average_clusterings_list]
    dd = pd.DataFrame(gossiper_opinions_list)
    dd.to_csv(Path + NetworkName + "gossiper_opinions" + ",media=" + str(config.num_media) + ",d_g=" + str(config.d_g) + "d_m=" + str(config.d_m) + ",y=" + str(
                config.Y_MEAN_TRAIN_EPISODES) + "," + str(Times) + ".csv")
    dd2 = pd.DataFrame(medias_opinion_list)
    dd2.to_csv(Path + NetworkName + "media_opinions" + ",media=" + str(config.num_media) + ",d_g=" + str(config.d_g) + "d_m=" + str(config.d_m) + ",y=" + str(
                config.Y_MEAN_TRAIN_EPISODES) + "," + str(Times) + ".csv")
    dd3 = pd.DataFrame(statistics)
    dd3.to_csv(Path + NetworkName + "statistics" + ",media=" + str(config.num_media) + ",d_g=" + str(config.d_g) + "d_m=" + str(config.d_m) + ",y=" + str(config.Y_MEAN_TRAIN_EPISODES) + "," + str(Times) + ".csv")


    for h in range(config.TRAIN_EPISODES):
        x = [h for _ in range(num_goss)]  # 正常情况
        plt.scatter(x, gossiper_opinions_list[h], color="black", marker='.')
        y = [h for _ in range(config.num_media)]
        plt.scatter(y, medias_opinion_list[h], color="red", marker='.')
        plt.tick_params(labelsize=12)
        plt.xlabel('Interaction Round', family='Calibri', weight='medium', size=18)
        plt.ylabel('Opinion', family='Calibri', weight='medium', size=18)
    plt.savefig(Path + NetworkName + ",media=" + str(config.num_media) + ",d_g=" + str(config.d_g) + "d_m=" + str(
        config.d_m) + ",y=" + str(
        config.Y_MEAN_TRAIN_EPISODES) + "," + str(Times) + ".png")
    plt.clf()


if __name__ == '__main__':

    NetworkName = "全连接"
    Path = "D:/fangdina/总结/代码2.0/wolf_cala/"
    config.num_media = 0
    for i in range(5):
        main(i, 0, 199, 0)
    config.num_media = 1
    for i in range(5):
        main(i, 1, 199, 0)
    config.num_media = 2
    for i in range(5):
        main(i, 2, 199, 0)
    config.num_media = 3
    for i in range(5):
        main(i, 3, 199, 0)
    config.num_media = 4
    for i in range(5):
        main(i, 4, 199, 0)
    config.num_media = 5
    for i in range(5):
        main(i, 5, 199, 0)



