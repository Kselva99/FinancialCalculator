import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_brownian_motion_model(S_0, K, days_to_maturity, r, sigma, iterations, contract_type):
    T = days_to_maturity / 365
    dt = T / days_to_maturity

    np.random.seed(0)
    S = np.zeros((days_to_maturity, iterations))
    S[0] = S_0

    for t in range(1, days_to_maturity):
        S[t] = S[t - 1] * np.exp(((r - (0.5 * sigma ** 2)) * dt) + (sigma * np.sqrt(dt) * np.random.standard_normal(iterations)))

    plt.figure(figsize=(12,12))
    plt.plot(S[:, :])
    plt.axhline(K, c="k", label="Strike Price")
    plt.ylabel("Simulated Price Movements")
    plt.xlabel("Days")
    plt.title("Monte Carlo Price Simulations")
    plt.legend(loc="best")
    plt.show()

    if(contract_type == "CALL"):
        return np.exp(-r * T) * (1 / iterations) * np.sum(np.maximum(S[-1] - K, 0))

    elif(contract_type == "PUT"):
        return np.exp(-r * T) * (1 / iterations) * np.sum(np.maximum(K - S[-1], 0))

    else:
        print("Invalid Option Contract Type (CALL or PUT)")
        return

S = 100
K = 110
days = 180
r = 0.05
sigma = 0.25
iterations = 100

mc_call = monte_carlo_brownian_motion_model(S, K, days, r, sigma, iterations, "CALL")
print(f"Monte Carlo Call Price: {mc_call}")
