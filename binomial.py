import numpy as np
from opt_pricing import Option_Pricing

class Binomial_Pricing(Option_Pricing):
    def __init__(self, spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, steps):
        super().__init__(spot_price, strike_price, days_to_maturity, risk_free_rate, 0, sigma, 'n/a')
        self.steps = steps
        self.ST = None
        self.call_price = None
        self.put_price = None

        self.generate()

        self.calc_call_price()
        self.calc_put_price()

    def generate(self):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt)) 
        d = 1 / u
        q = (np.exp((self.r) * dt) - d) / (u - d)

        # Initialize asset prices at maturity
        self.ST = np.zeros((self.steps + 1, self.steps + 1))
        self.ST[0, 0] = self.S

        for i in range(1, self.steps + 1):
            self.ST[i, 0] = self.ST[i - 1, 0] * u
            for j in range(1, i + 1):
                self.ST[i, j] = self.ST[i - 1, j - 1] * d

        # Initialize option values at maturity
        self.call_option_values = np.zeros((self.steps + 1, self.steps + 1))
        self.put_option_values = np.zeros((self.steps + 1, self.steps + 1))

        self.call_option_values[-1, :] = np.maximum(0, self.ST[-1, :] - self.K)
        self.put_option_values[-1, :] = np.maximum(0, self.K - self.ST[-1, :])

        # Step back through the tree
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                self.call_option_values[i, j] = np.exp(-self.r * dt) * (q * self.call_option_values[i + 1, j] + (1 - q) * self.call_option_values[i + 1, j + 1])
                self.put_option_values[i, j] = np.exp(-self.r * dt) * (q * self.put_option_values[i + 1, j] + (1 - q) * self.put_option_values[i + 1, j + 1])

    def calc_call_price(self):
        self.call_price = self.call_option_values[0, 0]

    def calc_put_price(self):
        self.put_price = self.put_option_values[0, 0]