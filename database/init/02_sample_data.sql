-- Sample data for development and testing
-- This script inserts sample data for development purposes

-- Insert sample trading strategy
INSERT INTO trading_strategies (
    name, 
    description, 
    configuration, 
    symbols
) VALUES (
    'IDTxl Causal Strategy',
    'Trading strategy based on information-theoretic causal relationships detected by IDTxl',
    '{
        "analysis_type": "transfer_entropy",
        "max_lag": 5,
        "estimator": "gaussian",
        "significance_level": 0.05,
        "position_sizing": "kelly_criterion",
        "risk_management": {
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.05
        }
    }',
    ARRAY['AAPL', 'MSFT', 'GOOGL']
) ON CONFLICT DO NOTHING;

-- Insert sample system health log
INSERT INTO system_health_logs (
    service_name,
    status,
    response_time
) VALUES (
    'backend_api',
    'healthy',
    0.05
) ON CONFLICT DO NOTHING;

-- Note: Time series data and analysis results will be populated
-- by the application during normal operation