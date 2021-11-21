import numpy as np
class config:
    num_media = 0
    d_g = 0.1
    d_m = 0.1  # 0.1
    TRAIN_EPISODES = 500  # 700
    Y_MEAN_TRAIN_EPISODES = 400  # 7014
    Dist = []
    delta = 0.001

    alpha_g = 0.5  # 学习率
    alpha_m = 0.5
    alpha_ub = 0.1 #0.05 #0.1  # 0.025
    alpha_us = 0.5 * alpha_ub #0.025 #0.5 * alpha_ub  # 0.25
    alpha_sigma =0.05 #0.01 #0.05  # 0.001
    ini_sigma = 1 / 6 #0.3#1 / 6
    gossiper_sample_num = 0.8  # 1 0.8 # 选择邻居或媒体的概率
    dist = 0

    u_medias = []
    for i in range(num_media):
        u_medias.append((i + 1) / (1 + num_media))
    y_mean = u_medias

    gossiper_opinion_restore = []
    media_opinion_restore = []
    r_list_1 = []
    r_list_2 = []
    sign = [False, False]

