import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from app.models.strategy import StrategyConfig, RiskManagementConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management service for trading strategies"""
    
    def __init__(self):
        self.risk_limits = {}
        self.position_limits = {}
        
    def validate_trade(
        self, 
        strategy_id: str,
        symbol: str, 
        trade_size: float, 
        current_portfolio: Dict[str, Any],
        risk_config: RiskManagementConfig
    ) -> Dict[str, Any]:
        """Validate a proposed trade against risk limits"""
        
        validation_result = {
            "approved": True,
            "warnings": [],
            "rejections": [],
            "suggested_size": trade_size
        }
        
        try:
            # Position size validation
            portfolio_value = current_portfolio.get('total_value', 100000)
            position_value = abs(trade_size)
            position_pct = position_value / portfolio_value
            
            if position_pct > risk_config.max_position_size:
                suggested_size = trade_size * (risk_config.max_position_size / position_pct)
                validation_result["warnings"].append(
                    f"Position size {position_pct:.2%} exceeds limit {risk_config.max_position_size:.2%}"
                )
                validation_result["suggested_size"] = suggested_size
            
            # Portfolio concentration check
            current_positions = current_portfolio.get('positions', {})
            if symbol in current_positions:
                new_position_value = abs(current_positions[symbol] + trade_size)
                new_position_pct = new_position_value / portfolio_value
                
                if new_position_pct > risk_config.max_position_size * 1.5:  # 150% of normal limit
                    validation_result["rejections"].append(
                        f"Combined position in {symbol} would be {new_position_pct:.2%}, exceeding concentration limit"
                    )
                    validation_result["approved"] = False
            
            # Daily loss limit check
            if risk_config.daily_loss_limit:
                daily_pnl = current_portfolio.get('daily_pnl', 0)
                daily_loss_pct = abs(min(0, daily_pnl)) / portfolio_value
                
                if daily_loss_pct >= risk_config.daily_loss_limit:
                    validation_result["rejections"].append(
                        f"Daily loss limit {risk_config.daily_loss_limit:.2%} already reached"
                    )
                    validation_result["approved"] = False
            
            # VaR limit check
            if risk_config.var_limit:
                current_var = self.calculate_portfolio_var(current_portfolio)
                if current_var > risk_config.var_limit:
                    validation_result["warnings"].append(
                        f"Portfolio VaR {current_var:.2%} exceeds limit {risk_config.var_limit:.2%}"
                    )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Trade validation failed: {str(e)}")
            return {
                "approved": False,
                "warnings": [],
                "rejections": [f"Validation error: {str(e)}"],
                "suggested_size": 0
            }
    
    def calculate_position_size(
        self,
        strategy_config: StrategyConfig,
        signal_strength: float,
        current_portfolio: Dict[str, Any],
        volatility: float
    ) -> float:
        """Calculate optimal position size based on risk management rules"""
        
        try:
            portfolio_value = current_portfolio.get('total_value', 100000)
            risk_config = strategy_config.risk_management
            
            # Base position size from max position limit
            base_size = portfolio_value * risk_config.max_position_size
            
            # Adjust based on position sizing method
            if strategy_config.execution_rules.position_sizing.value == "kelly_criterion":
                # Simplified Kelly Criterion
                win_rate = current_portfolio.get('historical_win_rate', 0.55)
                avg_win = current_portfolio.get('avg_win', 0.02)
                avg_loss = current_portfolio.get('avg_loss', -0.015)
                
                if avg_loss != 0:
                    kelly_fraction = (win_rate * avg_win + (1 - win_rate) * avg_loss) / abs(avg_loss)
                    kelly_fraction = max(0, min(kelly_fraction, risk_config.max_position_size))
                    base_size = portfolio_value * kelly_fraction
            
            elif strategy_config.execution_rules.position_sizing.value == "volatility_target":
                # Inverse volatility sizing
                target_vol = 0.02  # 2% daily volatility target
                vol_adjustment = target_vol / max(volatility, 0.005)  # Min volatility floor
                base_size *= vol_adjustment
            
            elif strategy_config.execution_rules.position_sizing.value == "risk_parity":
                # Equal risk contribution
                portfolio_volatility = current_portfolio.get('portfolio_volatility', 0.15)
                risk_budget = portfolio_value * 0.01  # 1% risk budget
                base_size = risk_budget / max(volatility, 0.005)
            
            # Adjust for signal strength
            adjusted_size = base_size * abs(signal_strength)
            
            # Apply direction
            final_size = adjusted_size * np.sign(signal_strength)
            
            # Ensure within limits
            max_size = portfolio_value * risk_config.max_position_size
            final_size = np.clip(final_size, -max_size, max_size)
            
            return final_size
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {str(e)}")
            return 0.0
    
    def calculate_portfolio_var(
        self, 
        portfolio: Dict[str, Any], 
        confidence_level: float = 0.95,
        holding_period: int = 1
    ) -> float:
        """Calculate portfolio Value at Risk"""
        
        try:
            positions = portfolio.get('positions', {})
            if not positions:
                return 0.0
            
            # Get historical returns for portfolio components
            # In a real implementation, this would use actual historical data
            
            # Simplified VaR calculation using portfolio volatility
            portfolio_value = portfolio.get('total_value', 100000)
            portfolio_volatility = portfolio.get('portfolio_volatility', 0.15)
            
            # Normal distribution assumption
            z_score = 1.645 if confidence_level == 0.95 else 2.326  # 95% or 99%
            daily_var = z_score * portfolio_volatility / np.sqrt(252)
            
            # Adjust for holding period
            var = daily_var * np.sqrt(holding_period)
            
            return var
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {str(e)}")
            return 0.0
    
    def check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_side: str,  # 'long' or 'short'
        stop_loss_pct: float
    ) -> bool:
        """Check if stop loss should be triggered"""
        
        if stop_loss_pct <= 0:
            return False
        
        if position_side == 'long':
            loss_pct = (entry_price - current_price) / entry_price
            return loss_pct >= stop_loss_pct
        else:  # short position
            loss_pct = (current_price - entry_price) / entry_price
            return loss_pct >= stop_loss_pct
    
    def check_take_profit(
        self,
        entry_price: float,
        current_price: float,
        position_side: str,
        take_profit_pct: float
    ) -> bool:
        """Check if take profit should be triggered"""
        
        if take_profit_pct <= 0:
            return False
        
        if position_side == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            return profit_pct >= take_profit_pct
        else:  # short position
            profit_pct = (entry_price - current_price) / entry_price
            return profit_pct >= take_profit_pct
    
    def calculate_drawdown(self, portfolio_values: List[float]) -> Dict[str, Any]:
        """Calculate current and maximum drawdown"""
        
        if len(portfolio_values) < 2:
            return {"current_drawdown": 0.0, "max_drawdown": 0.0}
        
        # Calculate running maximum
        peak = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown
        drawdown = (np.array(portfolio_values) - peak) / peak
        
        current_drawdown = drawdown[-1]
        max_drawdown = np.min(drawdown)
        
        return {
            "current_drawdown": abs(current_drawdown),
            "max_drawdown": abs(max_drawdown),
            "underwater_days": len([dd for dd in drawdown if dd < -0.01])
        }
    
    def check_risk_limits(
        self,
        strategy_id: str,
        current_portfolio: Dict[str, Any],
        risk_config: RiskManagementConfig
    ) -> Dict[str, Any]:
        """Check all risk limits for a strategy"""
        
        risk_status = {
            "within_limits": True,
            "limit_breaches": [],
            "warnings": [],
            "actions_required": []
        }
        
        try:
            portfolio_value = current_portfolio.get('total_value', 100000)
            
            # Check maximum drawdown
            portfolio_history = current_portfolio.get('value_history', [portfolio_value])
            drawdown_info = self.calculate_drawdown(portfolio_history)
            
            if drawdown_info['current_drawdown'] >= risk_config.max_drawdown:
                risk_status["within_limits"] = False
                risk_status["limit_breaches"].append(
                    f"Drawdown {drawdown_info['current_drawdown']:.2%} exceeds limit {risk_config.max_drawdown:.2%}"
                )
                risk_status["actions_required"].append("Consider reducing position sizes or stopping strategy")
            
            # Check VaR limit
            if risk_config.var_limit:
                current_var = self.calculate_portfolio_var(current_portfolio)
                if current_var >= risk_config.var_limit:
                    risk_status["warnings"].append(
                        f"VaR {current_var:.2%} approaching limit {risk_config.var_limit:.2%}"
                    )
            
            # Check daily loss limit
            if risk_config.daily_loss_limit:
                daily_pnl = current_portfolio.get('daily_pnl', 0)
                daily_loss_pct = abs(min(0, daily_pnl)) / portfolio_value
                
                if daily_loss_pct >= risk_config.daily_loss_limit:
                    risk_status["within_limits"] = False
                    risk_status["limit_breaches"].append(
                        f"Daily loss {daily_loss_pct:.2%} exceeds limit {risk_config.daily_loss_limit:.2%}"
                    )
                    risk_status["actions_required"].append("Stop trading for today")
            
            # Check position concentration
            positions = current_portfolio.get('positions', {})
            for symbol, position_value in positions.items():
                position_pct = abs(position_value) / portfolio_value
                if position_pct > risk_config.max_position_size * 1.2:  # 120% of limit
                    risk_status["warnings"].append(
                        f"Position in {symbol} ({position_pct:.2%}) exceeds recommended concentration"
                    )
            
            return risk_status
            
        except Exception as e:
            logger.error(f"Risk limit check failed: {str(e)}")
            return {
                "within_limits": False,
                "limit_breaches": [f"Risk check error: {str(e)}"],
                "warnings": [],
                "actions_required": ["Manual review required"]
            }
    
    def get_risk_report(
        self,
        strategy_id: str,
        current_portfolio: Dict[str, Any],
        risk_config: RiskManagementConfig
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        try:
            portfolio_value = current_portfolio.get('total_value', 100000)
            positions = current_portfolio.get('positions', {})
            
            # Portfolio metrics
            portfolio_var = self.calculate_portfolio_var(current_portfolio)
            portfolio_history = current_portfolio.get('value_history', [portfolio_value])
            drawdown_info = self.calculate_drawdown(portfolio_history)
            
            # Position analysis
            position_analysis = {}
            total_exposure = 0
            
            for symbol, position_value in positions.items():
                position_pct = abs(position_value) / portfolio_value
                total_exposure += position_pct
                
                position_analysis[symbol] = {
                    "value": position_value,
                    "percentage": position_pct,
                    "within_limit": position_pct <= risk_config.max_position_size,
                    "risk_contribution": position_pct * 0.15  # Simplified risk contribution
                }
            
            # Risk utilization
            risk_utilization = {
                "position_size_utilization": total_exposure / (risk_config.max_position_size * len(positions)) if positions else 0,
                "var_utilization": portfolio_var / risk_config.var_limit if risk_config.var_limit else 0,
                "drawdown_utilization": drawdown_info['current_drawdown'] / risk_config.max_drawdown
            }
            
            return {
                "strategy_id": strategy_id,
                "report_timestamp": datetime.utcnow().isoformat(),
                "portfolio_value": portfolio_value,
                "portfolio_var_95": portfolio_var,
                "current_drawdown": drawdown_info['current_drawdown'],
                "max_drawdown": drawdown_info['max_drawdown'],
                "total_exposure": total_exposure,
                "position_count": len(positions),
                "position_analysis": position_analysis,
                "risk_utilization": risk_utilization,
                "risk_limits": {
                    "max_position_size": risk_config.max_position_size,
                    "max_drawdown": risk_config.max_drawdown,
                    "var_limit": risk_config.var_limit,
                    "daily_loss_limit": risk_config.daily_loss_limit
                },
                "risk_status": self.check_risk_limits(strategy_id, current_portfolio, risk_config)
            }
            
        except Exception as e:
            logger.error(f"Risk report generation failed: {str(e)}")
            return {
                "strategy_id": strategy_id,
                "report_timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "risk_status": {"within_limits": False, "limit_breaches": [str(e)]}
            }