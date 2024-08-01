import numpy as np
from opt_pricing import Option_Pricing

# Implementation of Monte-Carlo simulation for European Options pricing
class Monte_Carlo_Pricing(Option_Pricing):
    def __init__(self, spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, iterations):
        super().__init__(spot_price, strike_price, days_to_maturity, risk_free_rate, 0, sigma, 'n/a')
        self.iter = iterations
        self.S_n = None
        self.call_price = None
        self.put_price = None

        self.simulate(days_to_maturity)

        self.calc_call_price()
        self.calc_put_price()

    def simulate(self, days):
        dt = self.T / days
        np.random.seed(0)

        self.S_n = np.zeros((days + 1, self.iter))
        self.S_n[0] = self.S

        for t in range(1, days + 1):
            self.S_n[t] = self.S_n[t - 1] * np.exp(((self.r - (0.5 * self.sigma ** 2)) * dt) + (self.sigma * np.sqrt(dt) * np.random.standard_normal(self.iter)))

    def calc_call_price(self):
        self.call_price = np.exp(-self.r * self.T) * (1 / self.iter) * np.sum(np.maximum(self.S_n[-1] - self.K, 0))

    def calc_put_price(self):
        self.put_price = np.exp(-self.r * self.T) * (1 / self.iter) * np.sum(np.maximum(self.K - self.S_n[-1], 0))