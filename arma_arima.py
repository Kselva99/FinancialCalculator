import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# # Data retrieval function
# def get_stock_data(ticker, start_date, end_date):
#     stock_data = yf.download(ticker, start=start_date, end=end_date)
#     return stock_data['Close']

# # Example usage
# ticker = 'AAPL'
# start_date = '2020-01-01'
# end_date = '2023-01-01'
# data = get_stock_data(ticker, start_date, end_date)
# data = data.dropna()

# # Scaling the data
# data_mean = data.mean()
# data_std = data.std()
# scaled_data = (data - data_mean) / data_std

# Define the ARMA model
class ARMA:
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def fit(self, data):
        self.data = data
        self.n = len(data)
        
        # Improved initial guess for parameters
        initial_params = np.random.randn(self.p + self.q + 1) * 0.1
        
        # Minimize the negative log-likelihood with bounds to ensure sigma > 0
        result = minimize(self._neg_log_likelihood, initial_params, method='L-BFGS-B', bounds=[(-np.inf, np.inf)]*(self.p + self.q) + [(1e-5, np.inf)])
        self.params = result.x

    def _neg_log_likelihood(self, params):
        phi = params[:self.p]
        theta = params[self.p:self.p + self.q]
        sigma = params[-1]
        
        # Calculate residuals
        residuals = self._get_residuals(phi, theta)
        
        # Regularization term to penalize extreme parameter values
        regularization = np.sum(params**2) * 0.01
        
        # Calculate log-likelihood
        ll = -0.5 * self.n * np.log(2 * np.pi * sigma ** 2) - 0.5 * np.sum(residuals ** 2) / (sigma ** 2) + regularization
        return -ll

    def _get_residuals(self, phi, theta):
        residuals = np.zeros(self.n)
        for t in range(max(self.p, self.q), self.n):
            ar_term = np.dot(phi, self.data[t-self.p:t][::-1])
            ma_term = np.dot(theta, residuals[t-self.q:t][::-1])
            residuals[t] = self.data[t] - ar_term - ma_term
        return residuals

    def predict(self, steps):
        predictions = np.zeros(steps)
        for t in range(steps):
            ar_term = np.dot(self.params[:self.p], self.data[-self.p:][::-1])
            ma_term = np.dot(self.params[self.p:self.p + self.q], predictions[-self.q:][::-1])
            predictions[t] = ar_term + ma_term
        return predictions

# # Example usage for ARMA
# arma_model = ARMA(p=2, q=2)
# arma_model.fit(scaled_data.values)
# arma_predictions = arma_model.predict(steps=10)

# # Scaling back the predictions
# arma_predictions = arma_predictions * data_std + data_mean

# print("ARMA Parameters:", arma_model.params)
# print("ARMA Predictions:", arma_predictions)

# Define the ARIMA model
class ARIMA:
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q

    def fit(self, data):
        self.original_data = data
        self.data = np.diff(data, n=self.d)
        self.n = len(self.data)
        
        # Improved initial guess for parameters
        initial_params = np.random.randn(self.p + self.q + 1) * 0.1
        
        # Minimize the negative log-likelihood with bounds to ensure sigma > 0
        result = minimize(self._neg_log_likelihood, initial_params, method='L-BFGS-B', bounds=[(-np.inf, np.inf)]*(self.p + self.q) + [(1e-5, np.inf)])
        self.params = result.x

    def _neg_log_likelihood(self, params):
        phi = params[:self.p]
        theta = params[self.p:self.p + self.q]
        sigma = params[-1]
        
        # Calculate residuals
        residuals = self._get_residuals(phi, theta)
        
        # Regularization term to penalize extreme parameter values
        regularization = np.sum(params**2) * 0.01
        
        # Calculate log-likelihood
        ll = -0.5 * self.n * np.log(2 * np.pi * sigma ** 2) - 0.5 * np.sum(residuals ** 2) / (sigma ** 2) + regularization
        return -ll

    def _get_residuals(self, phi, theta):
        residuals = np.zeros(self.n)
        for t in range(max(self.p, self.q), self.n):
            ar_term = np.dot(phi, self.data[t-self.p:t][::-1])
            ma_term = np.dot(theta, residuals[t-self.q:t][::-1])
            residuals[t] = self.data[t] - ar_term - ma_term
        return residuals

    def predict(self, steps):
        predictions = np.zeros(steps)
        for t in range(steps):
            ar_term = np.dot(self.params[:self.p], self.data[-self.p:][::-1])
            ma_term = np.dot(self.params[self.p:self.p + self.q], predictions[-self.q:][::-1])
            predictions[t] = ar_term + ma_term
        # Reintegrate the differencing
        predictions = np.cumsum(np.r_[self.original_data[-self.d:], predictions])
        return predictions[self.d:]

# # Example usage for ARIMA
# arima_model = ARIMA(p=2, d=1, q=2)
# arima_model.fit(scaled_data.values)
# arima_predictions = arima_model.predict(steps=10)

# # Scaling back the predictions
# arima_predictions = arima_predictions * data_std + data_mean

# print("ARIMA Parameters:", arima_model.params)
# print("ARIMA Predictions:", arima_predictions)
