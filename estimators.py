import numpy as np
from scipy import optimize
import math
from scipy.stats import norm


def calculation_d_1(s, k, r, t, σ):
    d_1 = (math.log(s / k) + (r + (σ**2) / 2) * t) / (math.sqrt(t) * σ)
    return d_1


def black_scholes_merton_european_call_option(s, k, r, t, σ):

    d_1 = calculation_d_1(s, k, r, t, σ)
    c = s * (norm.cdf(d_1)) - k * np.exp(-r * t) * (norm.cdf(d_1 - σ * math.sqrt(t)))

    return c


def black_scholes_merton_european_put_option(s, k, r, t, σ):
    d_1 = calculation_d_1(s, k, r, t, σ)
    p = k * np.exp(-r * t) * (norm.cdf(-(d_1 - σ * math.sqrt(t)))) - s * (
        norm.cdf(-d_1)
    )

    return p


def calculation_future(s, r, t):
    k = s * np.exp(r * t)

    return k


def calculation_pv(k, r, t):
    s = k * np.exp(-r * t)

    return s


def implied_volatility_european_call(s, k, r, t, c, a, b):

    def objective_function(sigma):
        return black_scholes_merton_european_call_option(s, k, r, t, sigma) - c

    x = optimize.brentq(objective_function, a, b)

    return x


def implied_volatility_european_put(s, k, r, t, p, a, b):
    def objective_function(sigma):
        return black_scholes_merton_european_put_option(s, k, r, t, sigma) - p

    x = optimize.brentq(objective_function, a, b)

    return x


def historical_volatility(s, t):
    u = np.log(s[1:] / s[:-1])
    var = np.std(u, ddof=1) / math.sqrt(t)

    return var
