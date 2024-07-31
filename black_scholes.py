import numpy as np
from scipy.stats import norm
from opt_pricing import Option_Pricing

# Implementation of Black Scholes model for European Options pricing 
class Black_Scholes_Pricing(Option_Pricing):
    def __init__(self, spot_price, strike_price, days_to_maturity, risk_free_rate, dividends, sigma, contract_type):
        super().__init__(spot_price, strike_price, days_to_maturity, risk_free_rate, dividends, sigma, contract_type)
        self.d1 = None
        self.d2 = None
        self.price = None

        if(self.contract_type == 'call'):
            self.calc_call_price()
            self.payoff = max(self.S - self.K, 0)
        elif(self.contract_type == 'put'):
            self.calc_put_price()
            self.payoff = max(self.K - self.S, 0)

    def calc_call_price(self):
        self.d1 = (np.log(self.S / self.K) + ((self.r - self.q + (0.5 * (self.sigma ** 2))) * self.T)) / (self.sigma * (self.T ** 0.5))
        self.d2 = (np.log(self.S / self.K) + ((self.r - self.q - (0.5 * (self.sigma ** 2))) * self.T)) / (self.sigma * (self.T ** 0.5))

        self.price = (self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)) - (self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))

    def calc_put_price(self):
        self.d1 = (np.log(self.S / self.K) + ((self.r - self.q + (0.5 * (self.sigma ** 2) * self.T)))) / (self.sigma * (self.T ** 0.5))
        self.d2 = (np.log(self.S / self.K) + ((self.r - self.q - (0.5 * (self.sigma ** 2) * self.T)))) / (self.sigma * (self.T ** 0.5))

        self.price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) - (self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))

    def calc_delta(self):
        if(self.contract_type == 'call'):
            self.delta = norm.cdf(self.d1)
        elif(self.contract_type == 'put'):
            self.delta = norm.cdf(self.d1) - 1
        return self.delta

    def calc_gamma(self):
        self.gamma = norm.pdf(self.d1) / (self.S * self.sigma * (self.T ** 0.5))
        return self.gamma

    def calc_theta(self):
        if(self.contract_type == 'call'):
            self.theta = ((-self.S * self.sigma * norm.pdf(self.d1)) / (2 * (self.T ** 0.5))) - (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        elif(self.contract_type == 'put'):
            self.theta = ((-self.S * self.sigma * norm.pdf(self.d1)) / (2 * (self.T ** 0.5))) + (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
        return self.theta

    def calc_vega(self):
        self.vega = self.S * (self.T ** 0.5) * norm.pdf(self.d1)
        return self.vega

    def calc_rho(self):
        if(self.contract_type == 'call'):
            self.rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        elif(self.contract_type == 'put'):
            self.rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return self.rho

    def gen_heatmap(self, min_spot, max_spot, min_vol, max_vol, gran=10):
        heat_spots = np.linspace(min_spot, max_spot, gran)
        heat_vols = np.linspace(min_vol, max_vol, gran)

        heatmap = np.zeros((gran, gran))

        for i in range(gran):
            cur_vol = heat_vols[i]
            tmp_d1 = (np.log(heat_spots / self.K) + ((self.r - self.q + (0.5 * (cur_vol ** 2))) * self.T)) / (cur_vol * (self.T ** 0.5))
            tmp_d2 = (np.log(heat_spots / self.K) + ((self.r - self.q - (0.5 * (cur_vol ** 2))) * self.T)) / (cur_vol * (self.T ** 0.5))
            
            if(self.contract_type == 'call'):
                heatmap[i] = (heat_spots * np.exp(-self.q * self.T) * norm.cdf(tmp_d1)) - (self.K * np.exp(-self.r * self.T) * norm.cdf(tmp_d2))
            elif(self.contract_type == 'put'):
                heatmap[i] = (self.K * np.exp(-self.r * self.T) * norm.cdf(-tmp_d2)) - (heat_spots * np.exp(-self.q * self.T) * norm.cdf(-tmp_d1))

        return heatmap, heat_spots, heat_vols