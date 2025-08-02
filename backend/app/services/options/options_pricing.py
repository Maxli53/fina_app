"""
Options Pricing Models Implementation
Includes Black-Scholes, Binomial, Monte Carlo, and Greeks calculations
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Option exercise style"""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class OptionContract:
    """Option contract specifications"""
    underlying: str
    strike: float
    expiry: datetime
    option_type: OptionType
    exercise_style: ExerciseStyle
    multiplier: int = 100
    dividend_yield: float = 0.0


@dataclass
class MarketData:
    """Market data for pricing"""
    spot_price: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0
    timestamp: datetime = None


@dataclass
class OptionGreeks:
    """Option Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho
        }


@dataclass
class PricingResult:
    """Options pricing result"""
    price: float
    greeks: OptionGreeks
    model: str
    timestamp: datetime
    additional_info: Dict[str, any] = None


class BlackScholesModel:
    """Black-Scholes options pricing model for European options"""
    
    @staticmethod
    def price(
        option: OptionContract,
        market: MarketData
    ) -> PricingResult:
        """Price European option using Black-Scholes formula"""
        
        # Calculate time to expiry in years
        time_to_expiry = (option.expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
        
        if time_to_expiry <= 0:
            # Option expired
            if option.option_type == OptionType.CALL:
                price = max(0, market.spot_price - option.strike)
            else:
                price = max(0, option.strike - market.spot_price)
            
            greeks = OptionGreeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
            
        else:
            # Calculate d1 and d2
            d1 = (np.log(market.spot_price / option.strike) + 
                  (market.risk_free_rate - market.dividend_yield + 0.5 * market.volatility**2) * time_to_expiry) / \
                 (market.volatility * np.sqrt(time_to_expiry))
            
            d2 = d1 - market.volatility * np.sqrt(time_to_expiry)
            
            # Calculate option price
            if option.option_type == OptionType.CALL:
                price = (market.spot_price * np.exp(-market.dividend_yield * time_to_expiry) * norm.cdf(d1) - 
                        option.strike * np.exp(-market.risk_free_rate * time_to_expiry) * norm.cdf(d2))
            else:  # PUT
                price = (option.strike * np.exp(-market.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                        market.spot_price * np.exp(-market.dividend_yield * time_to_expiry) * norm.cdf(-d1))
            
            # Calculate Greeks
            greeks = BlackScholesModel._calculate_greeks(
                option, market, d1, d2, time_to_expiry
            )
        
        return PricingResult(
            price=price,
            greeks=greeks,
            model="Black-Scholes",
            timestamp=datetime.now(),
            additional_info={
                "d1": d1 if time_to_expiry > 0 else None,
                "d2": d2 if time_to_expiry > 0 else None,
                "time_to_expiry": time_to_expiry
            }
        )
    
    @staticmethod
    def _calculate_greeks(
        option: OptionContract,
        market: MarketData,
        d1: float,
        d2: float,
        time_to_expiry: float
    ) -> OptionGreeks:
        """Calculate option Greeks"""
        
        # Delta
        if option.option_type == OptionType.CALL:
            delta = np.exp(-market.dividend_yield * time_to_expiry) * norm.cdf(d1)
        else:
            delta = -np.exp(-market.dividend_yield * time_to_expiry) * norm.cdf(-d1)
        
        # Gamma (same for calls and puts)
        gamma = (np.exp(-market.dividend_yield * time_to_expiry) * norm.pdf(d1)) / \
                (market.spot_price * market.volatility * np.sqrt(time_to_expiry))
        
        # Theta
        if option.option_type == OptionType.CALL:
            theta = (-market.spot_price * np.exp(-market.dividend_yield * time_to_expiry) * 
                    norm.pdf(d1) * market.volatility / (2 * np.sqrt(time_to_expiry)) +
                    market.dividend_yield * market.spot_price * np.exp(-market.dividend_yield * time_to_expiry) * norm.cdf(d1) -
                    market.risk_free_rate * option.strike * np.exp(-market.risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:
            theta = (-market.spot_price * np.exp(-market.dividend_yield * time_to_expiry) * 
                    norm.pdf(d1) * market.volatility / (2 * np.sqrt(time_to_expiry)) -
                    market.dividend_yield * market.spot_price * np.exp(-market.dividend_yield * time_to_expiry) * norm.cdf(-d1) +
                    market.risk_free_rate * option.strike * np.exp(-market.risk_free_rate * time_to_expiry) * norm.cdf(-d2))
        
        # Convert theta to per day
        theta = theta / 365.25
        
        # Vega (same for calls and puts)
        vega = market.spot_price * np.exp(-market.dividend_yield * time_to_expiry) * \
               norm.pdf(d1) * np.sqrt(time_to_expiry) / 100  # Per 1% change in volatility
        
        # Rho
        if option.option_type == OptionType.CALL:
            rho = option.strike * time_to_expiry * \
                  np.exp(-market.risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100  # Per 1% change in rate
        else:
            rho = -option.strike * time_to_expiry * \
                  np.exp(-market.risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
        
        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )
    
    @staticmethod
    def implied_volatility(
        option: OptionContract,
        market_price: float,
        market: MarketData,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        
        # Initial guess
        vol = 0.3
        
        for _ in range(max_iterations):
            # Create temporary market data with current vol estimate
            temp_market = MarketData(
                spot_price=market.spot_price,
                risk_free_rate=market.risk_free_rate,
                volatility=vol,
                dividend_yield=market.dividend_yield
            )
            
            # Price option
            result = BlackScholesModel.price(option, temp_market)
            
            # Calculate error
            price_error = result.price - market_price
            
            if abs(price_error) < tolerance:
                return vol
            
            # Update volatility estimate using vega
            if result.greeks.vega != 0:
                vol = vol - price_error / (result.greeks.vega * 100)
                vol = max(0.001, min(5.0, vol))  # Keep within reasonable bounds
            else:
                break
        
        # If Newton-Raphson fails, try Brent's method
        try:
            def objective(v):
                temp_market = MarketData(
                    spot_price=market.spot_price,
                    risk_free_rate=market.risk_free_rate,
                    volatility=v,
                    dividend_yield=market.dividend_yield
                )
                return BlackScholesModel.price(option, temp_market).price - market_price
            
            vol = brentq(objective, 0.001, 5.0, xtol=tolerance)
            return vol
        except:
            logger.warning(f"Failed to calculate implied volatility for {option.underlying}")
            return np.nan


class BinomialModel:
    """Binomial tree model for American and European options"""
    
    @staticmethod
    def price(
        option: OptionContract,
        market: MarketData,
        steps: int = 100
    ) -> PricingResult:
        """Price option using binomial tree"""
        
        # Calculate parameters
        time_to_expiry = (option.expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
        
        if time_to_expiry <= 0:
            # Option expired
            if option.option_type == OptionType.CALL:
                price = max(0, market.spot_price - option.strike)
            else:
                price = max(0, option.strike - market.spot_price)
            
            greeks = OptionGreeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
            
        else:
            dt = time_to_expiry / steps
            u = np.exp(market.volatility * np.sqrt(dt))  # Up move
            d = 1 / u  # Down move
            p = (np.exp((market.risk_free_rate - market.dividend_yield) * dt) - d) / (u - d)  # Risk-neutral probability
            
            # Build stock price tree
            stock_tree = np.zeros((steps + 1, steps + 1))
            stock_tree[0, 0] = market.spot_price
            
            for i in range(1, steps + 1):
                stock_tree[i, 0] = stock_tree[i-1, 0] * u
                for j in range(1, i + 1):
                    stock_tree[i, j] = stock_tree[i-1, j-1] * d
            
            # Calculate option values at expiry
            option_tree = np.zeros((steps + 1, steps + 1))
            
            if option.option_type == OptionType.CALL:
                option_tree[steps] = np.maximum(0, stock_tree[steps] - option.strike)
            else:
                option_tree[steps] = np.maximum(0, option.strike - stock_tree[steps])
            
            # Backward induction
            for i in range(steps - 1, -1, -1):
                for j in range(i + 1):
                    # Discounted expected value
                    option_tree[i, j] = np.exp(-market.risk_free_rate * dt) * \
                                       (p * option_tree[i+1, j] + (1-p) * option_tree[i+1, j+1])
                    
                    # For American options, check early exercise
                    if option.exercise_style == ExerciseStyle.AMERICAN:
                        if option.option_type == OptionType.CALL:
                            exercise_value = stock_tree[i, j] - option.strike
                        else:
                            exercise_value = option.strike - stock_tree[i, j]
                        
                        option_tree[i, j] = max(option_tree[i, j], exercise_value)
            
            price = option_tree[0, 0]
            
            # Calculate Greeks using finite differences
            greeks = BinomialModel._calculate_greeks_fd(
                option, market, price, steps
            )
        
        return PricingResult(
            price=price,
            greeks=greeks,
            model="Binomial",
            timestamp=datetime.now(),
            additional_info={
                "steps": steps,
                "time_to_expiry": time_to_expiry
            }
        )
    
    @staticmethod
    def _calculate_greeks_fd(
        option: OptionContract,
        market: MarketData,
        base_price: float,
        steps: int
    ) -> OptionGreeks:
        """Calculate Greeks using finite differences"""
        
        # Delta - change in price for 1% move in spot
        spot_bump = market.spot_price * 0.01
        
        market_up = MarketData(
            spot_price=market.spot_price + spot_bump,
            risk_free_rate=market.risk_free_rate,
            volatility=market.volatility,
            dividend_yield=market.dividend_yield
        )
        price_up = BinomialModel.price(option, market_up, steps).price
        
        market_down = MarketData(
            spot_price=market.spot_price - spot_bump,
            risk_free_rate=market.risk_free_rate,
            volatility=market.volatility,
            dividend_yield=market.dividend_yield
        )
        price_down = BinomialModel.price(option, market_down, steps).price
        
        delta = (price_up - price_down) / (2 * spot_bump)
        
        # Gamma - rate of change of delta
        gamma = (price_up - 2 * base_price + price_down) / (spot_bump ** 2)
        
        # Vega - change for 1% move in volatility
        vol_bump = 0.01
        market_vol_up = MarketData(
            spot_price=market.spot_price,
            risk_free_rate=market.risk_free_rate,
            volatility=market.volatility + vol_bump,
            dividend_yield=market.dividend_yield
        )
        price_vol_up = BinomialModel.price(option, market_vol_up, steps).price
        vega = (price_vol_up - base_price) / 100  # Per 1% change
        
        # Theta - time decay (price tomorrow vs today)
        # Approximate by pricing with slightly less time
        option_tomorrow = OptionContract(
            underlying=option.underlying,
            strike=option.strike,
            expiry=option.expiry - timedelta(days=1),
            option_type=option.option_type,
            exercise_style=option.exercise_style
        )
        price_tomorrow = BinomialModel.price(option_tomorrow, market, steps).price
        theta = price_tomorrow - base_price
        
        # Rho - change for 1% move in interest rate
        rate_bump = 0.01
        market_rate_up = MarketData(
            spot_price=market.spot_price,
            risk_free_rate=market.risk_free_rate + rate_bump,
            volatility=market.volatility,
            dividend_yield=market.dividend_yield
        )
        price_rate_up = BinomialModel.price(option, market_rate_up, steps).price
        rho = (price_rate_up - base_price) / 100  # Per 1% change
        
        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )


class MonteCarloModel:
    """Monte Carlo simulation for exotic and path-dependent options"""
    
    @staticmethod
    def price(
        option: OptionContract,
        market: MarketData,
        simulations: int = 10000,
        time_steps: int = 252,
        antithetic: bool = True,
        control_variate: bool = True
    ) -> PricingResult:
        """Price option using Monte Carlo simulation"""
        
        # Calculate parameters
        time_to_expiry = (option.expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
        
        if time_to_expiry <= 0:
            # Option expired
            if option.option_type == OptionType.CALL:
                price = max(0, market.spot_price - option.strike)
            else:
                price = max(0, option.strike - market.spot_price)
            
            greeks = OptionGreeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
            
        else:
            dt = time_to_expiry / time_steps
            
            # Generate random paths
            if antithetic:
                # Use antithetic variates for variance reduction
                randn = np.random.standard_normal((simulations // 2, time_steps))
                randn = np.concatenate([randn, -randn], axis=0)
            else:
                randn = np.random.standard_normal((simulations, time_steps))
            
            # Simulate price paths
            drift = (market.risk_free_rate - market.dividend_yield - 0.5 * market.volatility**2) * dt
            diffusion = market.volatility * np.sqrt(dt)
            
            price_paths = np.zeros((simulations, time_steps + 1))
            price_paths[:, 0] = market.spot_price
            
            for t in range(1, time_steps + 1):
                price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * randn[:, t-1])
            
            # Calculate payoffs
            if option.option_type == OptionType.CALL:
                payoffs = np.maximum(0, price_paths[:, -1] - option.strike)
            else:
                payoffs = np.maximum(0, option.strike - price_paths[:, -1])
            
            # Handle American options
            if option.exercise_style == ExerciseStyle.AMERICAN:
                payoffs = MonteCarloModel._american_option_payoff(
                    option, price_paths, market, time_to_expiry
                )
            
            # Apply control variate if requested
            if control_variate and option.exercise_style == ExerciseStyle.EUROPEAN:
                # Use Black-Scholes as control
                bs_result = BlackScholesModel.price(option, market)
                bs_price = bs_result.price
                
                # Calculate control variate adjustment
                if option.option_type == OptionType.CALL:
                    control_payoffs = np.maximum(0, price_paths[:, -1] - option.strike)
                else:
                    control_payoffs = np.maximum(0, option.strike - price_paths[:, -1])
                
                control_price = np.exp(-market.risk_free_rate * time_to_expiry) * np.mean(control_payoffs)
                
                # Adjust payoffs
                beta = np.cov(payoffs, control_payoffs)[0, 1] / np.var(control_payoffs)
                payoffs = payoffs - beta * (control_price - bs_price)
            
            # Discount to present value
            price = np.exp(-market.risk_free_rate * time_to_expiry) * np.mean(payoffs)
            
            # Estimate standard error
            std_error = np.std(payoffs) / np.sqrt(simulations)
            
            # Calculate Greeks using pathwise derivatives
            greeks = MonteCarloModel._calculate_greeks_pathwise(
                option, market, price_paths, time_to_expiry
            )
        
        return PricingResult(
            price=price,
            greeks=greeks,
            model="Monte Carlo",
            timestamp=datetime.now(),
            additional_info={
                "simulations": simulations,
                "time_steps": time_steps,
                "standard_error": std_error if time_to_expiry > 0 else 0,
                "antithetic": antithetic,
                "control_variate": control_variate
            }
        )
    
    @staticmethod
    def _american_option_payoff(
        option: OptionContract,
        price_paths: np.ndarray,
        market: MarketData,
        time_to_expiry: float
    ) -> np.ndarray:
        """Calculate American option payoff using Longstaff-Schwartz method"""
        
        simulations, time_steps = price_paths.shape[0], price_paths.shape[1] - 1
        dt = time_to_expiry / time_steps
        
        # Initialize cash flows
        cash_flows = np.zeros((simulations, time_steps + 1))
        
        # Calculate exercise value at each time step
        if option.option_type == OptionType.CALL:
            exercise_values = np.maximum(0, price_paths - option.strike)
        else:
            exercise_values = np.maximum(0, option.strike - price_paths)
        
        # Set terminal payoff
        cash_flows[:, -1] = exercise_values[:, -1]
        
        # Backward induction
        for t in range(time_steps - 1, 0, -1):
            # Find in-the-money paths
            itm = exercise_values[:, t] > 0
            
            if np.sum(itm) > 0:
                # Regression to estimate continuation value
                X = price_paths[itm, t]
                Y = np.exp(-market.risk_free_rate * dt) * cash_flows[itm, t+1]
                
                # Use polynomial basis functions
                A = np.vstack([np.ones_like(X), X, X**2]).T
                coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
                
                # Estimate continuation value
                continuation_value = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2
                
                # Exercise if immediate exercise value > continuation value
                exercise = exercise_values[itm, t] > continuation_value
                
                # Update cash flows
                exercise_indices = np.where(itm)[0][exercise]
                cash_flows[exercise_indices, t] = exercise_values[exercise_indices, t]
                cash_flows[exercise_indices, t+1:] = 0
        
        # Discount cash flows to present
        discount_factors = np.exp(-market.risk_free_rate * np.arange(time_steps + 1) * dt)
        return np.sum(cash_flows * discount_factors, axis=1)
    
    @staticmethod
    def _calculate_greeks_pathwise(
        option: OptionContract,
        market: MarketData,
        price_paths: np.ndarray,
        time_to_expiry: float
    ) -> OptionGreeks:
        """Calculate Greeks using pathwise method"""
        
        simulations = price_paths.shape[0]
        final_prices = price_paths[:, -1]
        
        # Delta using pathwise derivative
        if option.option_type == OptionType.CALL:
            indicator = final_prices > option.strike
        else:
            indicator = final_prices < option.strike
        
        delta = np.exp(-market.risk_free_rate * time_to_expiry) * \
                np.mean(indicator * final_prices / market.spot_price)
        
        # Gamma using likelihood ratio method
        gamma = np.exp(-market.risk_free_rate * time_to_expiry) * \
                np.mean(indicator * (np.log(final_prices / market.spot_price) - 
                       (market.risk_free_rate - 0.5 * market.volatility**2) * time_to_expiry) / 
                       (market.spot_price * market.volatility**2 * time_to_expiry))
        
        # Vega using pathwise derivative
        vega = np.exp(-market.risk_free_rate * time_to_expiry) * \
               np.mean(indicator * final_prices * 
                      (np.log(final_prices / market.spot_price) - 
                       (market.risk_free_rate + 0.5 * market.volatility**2) * time_to_expiry) / 
                      market.volatility) / 100
        
        # Theta and Rho using finite differences (simplified)
        theta = 0  # Would require re-simulation
        rho = 0    # Would require re-simulation
        
        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )


class VolatilitySurface:
    """Volatility surface modeling and interpolation"""
    
    def __init__(self, strikes: List[float], expiries: List[float], vols: np.ndarray):
        self.strikes = np.array(strikes)
        self.expiries = np.array(expiries)
        self.vols = vols
        
    def get_vol(self, strike: float, expiry: float) -> float:
        """Interpolate volatility for given strike and expiry"""
        from scipy.interpolate import griddata
        
        # Create mesh
        strike_mesh, expiry_mesh = np.meshgrid(self.strikes, self.expiries)
        points = np.column_stack((strike_mesh.ravel(), expiry_mesh.ravel()))
        values = self.vols.ravel()
        
        # Interpolate
        vol = griddata(points, values, (strike, expiry), method='cubic')
        
        return float(vol)
    
    def calibrate_sabr(self, spot: float, rate: float) -> Dict[str, float]:
        """Calibrate SABR model to volatility surface"""
        # Simplified SABR calibration
        # In practice, use more sophisticated calibration
        
        params = {
            "alpha": 0.3,
            "beta": 0.5,
            "rho": -0.3,
            "nu": 0.4
        }
        
        return params


class OptionsPricingService:
    """Main service for options pricing"""
    
    def __init__(self):
        self.models = {
            "black-scholes": BlackScholesModel,
            "binomial": BinomialModel,
            "monte-carlo": MonteCarloModel
        }
        
    def price_option(
        self,
        option: OptionContract,
        market: MarketData,
        model: str = "black-scholes",
        **kwargs
    ) -> PricingResult:
        """Price option using specified model"""
        
        if model not in self.models:
            raise ValueError(f"Unknown model: {model}")
        
        # Use Black-Scholes for European options by default
        if model == "black-scholes" and option.exercise_style == ExerciseStyle.AMERICAN:
            logger.info("Black-Scholes doesn't support American options, using Binomial model")
            model = "binomial"
        
        # Get model class
        model_class = self.models[model]
        
        # Price option
        return model_class.price(option, market, **kwargs)
    
    def calculate_implied_volatility(
        self,
        option: OptionContract,
        market_price: float,
        market: MarketData
    ) -> float:
        """Calculate implied volatility"""
        
        if option.exercise_style == ExerciseStyle.EUROPEAN:
            return BlackScholesModel.implied_volatility(option, market_price, market)
        else:
            # Use iterative approach for American options
            return self._implied_vol_american(option, market_price, market)
    
    def _implied_vol_american(
        self,
        option: OptionContract,
        market_price: float,
        market: MarketData,
        tolerance: float = 1e-4
    ) -> float:
        """Calculate implied volatility for American options"""
        
        def objective(vol):
            temp_market = MarketData(
                spot_price=market.spot_price,
                risk_free_rate=market.risk_free_rate,
                volatility=vol,
                dividend_yield=market.dividend_yield
            )
            result = self.price_option(option, temp_market, model="binomial", steps=100)
            return result.price - market_price
        
        try:
            # Use Brent's method
            vol = brentq(objective, 0.001, 5.0, xtol=tolerance)
            return vol
        except:
            logger.warning(f"Failed to calculate implied volatility for American option {option.underlying}")
            return np.nan
    
    def price_portfolio(
        self,
        options: List[Tuple[OptionContract, int]],  # (option, quantity)
        market: MarketData,
        model: str = "black-scholes"
    ) -> Dict[str, Any]:
        """Price portfolio of options"""
        
        total_value = 0
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        positions = []
        
        for option, quantity in options:
            result = self.price_option(option, market, model)
            
            position_value = result.price * quantity * option.multiplier
            total_value += position_value
            
            # Aggregate Greeks
            total_delta += result.greeks.delta * quantity * option.multiplier
            total_gamma += result.greeks.gamma * quantity * option.multiplier
            total_theta += result.greeks.theta * quantity * option.multiplier
            total_vega += result.greeks.vega * quantity * option.multiplier
            total_rho += result.greeks.rho * quantity * option.multiplier
            
            positions.append({
                "option": option,
                "quantity": quantity,
                "price": result.price,
                "value": position_value,
                "greeks": result.greeks.to_dict()
            })
        
        return {
            "total_value": total_value,
            "portfolio_greeks": {
                "delta": total_delta,
                "gamma": total_gamma,
                "theta": total_theta,
                "vega": total_vega,
                "rho": total_rho
            },
            "positions": positions,
            "timestamp": datetime.now()
        }
    
    def calculate_var_options(
        self,
        options: List[Tuple[OptionContract, int]],
        market: MarketData,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        simulations: int = 10000
    ) -> Dict[str, float]:
        """Calculate VaR for options portfolio"""
        
        # Generate market scenarios
        returns = np.random.normal(
            0, 
            market.volatility * np.sqrt(horizon_days / 252),
            simulations
        )
        
        vol_changes = np.random.normal(0, 0.01, simulations)  # 1% daily vol change
        
        portfolio_values = []
        
        for i in range(simulations):
            # Create scenario market data
            scenario_market = MarketData(
                spot_price=market.spot_price * (1 + returns[i]),
                risk_free_rate=market.risk_free_rate,
                volatility=max(0.01, market.volatility + vol_changes[i]),
                dividend_yield=market.dividend_yield
            )
            
            # Price portfolio under scenario
            portfolio = self.price_portfolio(options, scenario_market)
            portfolio_values.append(portfolio["total_value"])
        
        # Calculate current portfolio value
        current_portfolio = self.price_portfolio(options, market)
        current_value = current_portfolio["total_value"]
        
        # Calculate P&L distribution
        pnl = np.array(portfolio_values) - current_value
        
        # Calculate VaR
        var = -np.percentile(pnl, (1 - confidence_level) * 100)
        
        # Calculate CVaR (Expected Shortfall)
        cvar = -np.mean(pnl[pnl < -var])
        
        return {
            "var": var,
            "cvar": cvar,
            "confidence_level": confidence_level,
            "horizon_days": horizon_days,
            "current_value": current_value,
            "worst_loss": -np.min(pnl),
            "best_gain": np.max(pnl)
        }