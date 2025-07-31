-- Financial Time Series Analysis Platform Database Schema
-- This script initializes the database tables for the platform

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Time series data storage
CREATE TABLE IF NOT EXISTS time_series_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open_price DECIMAL(15, 6) NOT NULL,
    high_price DECIMAL(15, 6) NOT NULL,
    low_price DECIMAL(15, 6) NOT NULL,
    close_price DECIMAL(15, 6) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close DECIMAL(15, 6),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, interval, timestamp)
);

-- Analysis results storage
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_type VARCHAR(50) NOT NULL,
    symbols TEXT[] NOT NULL,
    configuration JSONB NOT NULL,
    results JSONB NOT NULL,
    processing_time DECIMAL(10, 3),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'completed'
);

-- Data quality reports
CREATE TABLE IF NOT EXISTS data_quality_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    period_start TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    period_end TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    completeness DECIMAL(5, 4) NOT NULL,
    consistency DECIMAL(5, 4) NOT NULL,
    timeliness DECIMAL(5, 4) NOT NULL,
    accuracy DECIMAL(5, 4) NOT NULL,
    issues TEXT[],
    recommendations TEXT[],
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Trading strategies
CREATE TABLE IF NOT EXISTS trading_strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    configuration JSONB NOT NULL,
    symbols TEXT[] NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID REFERENCES trading_strategies(id),
    configuration JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    trades JSONB,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System health and monitoring
CREATE TABLE IF NOT EXISTS system_health_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_time DECIMAL(10, 3),
    error_message TEXT,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_time_series_symbol_interval ON time_series_data(symbol, interval);
CREATE INDEX IF NOT EXISTS idx_time_series_timestamp ON time_series_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_analysis_results_type ON analysis_results(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at);
CREATE INDEX IF NOT EXISTS idx_data_quality_symbol ON data_quality_reports(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_strategies_active ON trading_strategies(is_active);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health_logs(timestamp);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at
CREATE TRIGGER update_time_series_data_updated_at BEFORE UPDATE ON time_series_data
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_strategies_updated_at BEFORE UPDATE ON trading_strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();