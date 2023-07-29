import numpy as np
from scipy.stats import norm

def black_scholes_model(S, K, T, r, sigma, contract_type):
    d1 = (np.log(S / K) + ((r + (0.5 * sigma ** 2)) * T)) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + ((r - (0.5 * sigma ** 2)) * T)) / (sigma * np.sqrt(T))
    
    if(contract_type == "CALL"):
        return (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))

    elif(contract_type == "PUT"):
        return (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    
    else:
        print("Invalid Option Contract Type (CALL or PUT)")
        return

S = 100
K = 110
days = 180
r = 0.05
sigma = 0.25
iterations = 100

bs_call = black_scholes_model(S, K, days / 365, r, sigma, "CALL")
print(f"Black Scholes Call Price: {bs_call}")