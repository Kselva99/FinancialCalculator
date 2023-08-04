from abc import ABC, abstractmethod

# Abstract class to price call and put options
class Option_Pricing(ABC):
    def __init__(self, spot_price, strike_price, days_to_maturity, risk_free_rate, dividends, sigma, contract_type):
        self.S = spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.q = dividends
        self.sigma = sigma
        
        self.contract_type = contract_type.lower()
        self.price = 0
        super().__init__()

    @abstractmethod
    def calc_call_price(self):
        pass

    @abstractmethod
    def calc_put_price(self):
        pass