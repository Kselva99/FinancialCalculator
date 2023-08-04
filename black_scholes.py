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

    def calc_gamma(self):
        self.gamma = norm.pdf(self.d1) / (self.S * self.sigma * (self.T ** 0.5))

    def calc_theta(self):
        if(self.contract_type == 'call'):
            self.theta = ((-self.S * self.sigma * norm.pdf(self.d1)) / (2 * (self.T ** 0.5))) - (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        elif(self.contract_type == 'put'):
            self.theta = ((-self.S * self.sigma * norm.pdf(self.d1)) / (2 * (self.T ** 0.5))) + (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))

    def calc_vega(self):
        self.vega = self.S * (self.T ** 0.5) * norm.pdf(self.d1)

    def calc_rho(self):
        if(self.contract_type == 'call'):
            self.rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        elif(self.contract_type == 'put'):
            self.rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)