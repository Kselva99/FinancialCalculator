import numpy as np
from opt_pricing import Option_Pricing

class Trinomial_Pricing(Option_Pricing):
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
        u = np.exp(self.sigma * np.sqrt(2 * dt))
        d = 1 / u
        m = 1 

        pu = ((np.exp((self.r) * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) / (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))) ** 2
        pd = ((np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp((self.r) * dt / 2)) / (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))) ** 2
        pm = 1 - pu - pd 

        # Initialize asset prices at maturity
        self.ST = np.zeros((2 * self.steps + 1, self.steps + 1))
        self.ST[self.steps, 0] = self.S

        for i in range(1, self.steps + 1):
            for j in range(2 * i + 1):
                if j == 0:
                    self.ST[self.steps - i + j, i] = self.ST[self.steps - i + j + 1, i - 1] * d
                elif j == 2 * i:
                    self.ST[self.steps - i + j, i] = self.ST[self.steps - i + j - 1, i - 1] * u
                else:
                    self.ST[self.steps - i + j, i] = self.ST[self.steps - i + j, i - 1] * m

        # Initialize option values at maturity
        self.call_option_values = np.zeros((2 * self.steps + 1, self.steps + 1))
        self.put_option_values = np.zeros((2 * self.steps + 1, self.steps + 1))

        self.call_option_values[:, -1] = np.maximum(0, self.ST[:, -1] - self.K)
        self.put_option_values[:, -1] = np.maximum(0, self.K - self.ST[:, -1])

        # Step back through the tree
        for i in range(self.steps - 1, -1, -1):
            for j in range(2 * i + 1):
                call_value = 0
                put_value = 0

                if self.steps - i + j - 1 >= 0 and self.steps - i + j - 1 < 2 * self.steps + 1:
                    call_value += pu * self.call_option_values[self.steps - i + j - 1, i + 1]
                    put_value += pu * self.put_option_values[self.steps - i + j - 1, i + 1]

                if self.steps - i + j < 2 * self.steps + 1:
                    call_value += pm * self.call_option_values[self.steps - i + j, i + 1]
                    put_value += pm * self.put_option_values[self.steps - i + j, i + 1]

                if self.steps - i + j + 1 < 2 * self.steps + 1:
                    call_value += pd * self.call_option_values[self.steps - i + j + 1, i + 1]
                    put_value += pd * self.put_option_values[self.steps - i + j + 1, i + 1]

                self.call_option_values[self.steps - i + j, i] = np.exp(-self.r * dt) * call_value
                self.put_option_values[self.steps - i + j, i] = np.exp(-self.r * dt) * put_value

    def calc_call_price(self):
        self.call_price = self.call_option_values[self.steps, 0]

    def calc_put_price(self):
        self.put_price = self.put_option_values[self.steps, 0]
