class Update_strategy(object):
    def __init__(self, deta, u_ini, sigma_ini):
        self.d_m = config.d_m
        self.d_g = config.d_g
        self.alpha_sigma = config.alpha_sigma
        self.alpha_ub = config.alpha_ub
        self.alpha_us = config.alpha_us
        self.deta = deta

        self.Q = 0
        self.y_mean = 0
        self.c = 0
        self.sigma_ini = sigma_ini
        self.u = u_ini
        self.sigma = sigma_ini