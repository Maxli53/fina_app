# AI Advisory System Documentation

## Overview

The Financial Platform integrates a PhD-level AI advisory system powered by GPT-4 (with Claude as an alternative) to provide expert guidance on quantitative analysis, trading strategies, risk management, and portfolio optimization.

## Architecture

```
AI Advisory System
├── GPT Advisor Service
│   ├── Multiple Advisory Roles
│   ├── Context Management (with full configuration awareness)
│   └── Response Parsing (extracts optimal parameters)
├── API Endpoints
│   ├── Analysis Configuration (dynamic, context-aware)
│   ├── Results Interpretation
│   └── Interactive Consultation
└── Frontend Integration
    ├── Chat Interface
    ├── Analysis Assistant (real-time AI recommendations)
    └── Strategy Builder
```

## Recent Improvements (August 2025)

### Dynamic AI Recommendations
The AI Advisory system now provides **true PhD-level expertise** instead of hardcoded rules:

1. **Full Context Awareness**: AI receives complete configuration details including all parameter values
2. **Real-time Analysis**: Configuration changes trigger immediate AI analysis
3. **Parameter Optimization**: AI suggests optimal values based on specific dataset and objectives
4. **No More Hardcoded Rules**: All recommendations come from actual AI analysis

## Advisory Roles

### 1. Quantitative Analyst

**Expertise:**
- Information theory and transfer entropy analysis (IDTxl)
- Advanced statistical methods and time series analysis
- Machine learning for financial markets
- Neural network architectures for price prediction
- Cross-asset correlation and causality analysis

**Use Cases:**
```python
# Get analysis recommendations with full configuration context
response = await advisor.get_analysis_recommendations(
    context=AnalysisContext(
        symbols=["AAPL", "MSFT", "GOOGL"],
        timeframe="1d",
        analysis_type="idtxl",
        objectives=["optimize_configuration", "maximize_signal_quality"],
        risk_tolerance="moderate",
        capital=100000,
        additional_context={
            "active_analysis": "idtxl",
            "configuration": {
                "estimator": "kraskov",
                "maxLag": 5,
                "permutations": 200,
                "significanceLevel": 0.05,
                "kNeighbors": 4
            }
        }
    ),
    role=AdvisorRole.QUANT_ANALYST
)

# AI will analyze the specific configuration and return:
# - Recommendations: "For significance level 0.05, increase permutations to at least 500"
# - Insights: "Kraskov estimator with k=4 may underestimate MI, consider k=5-7"
# - Optimal config: {"permutations": 500, "kNeighbors": 6}
```

### 2. Risk Manager

**Expertise:**
- Value at Risk (VaR) and Conditional VaR
- Portfolio risk decomposition
- Stress testing and scenario analysis
- Correlation breakdowns and tail risk
- Regulatory compliance (Basel III, MiFID II)

**Example Analysis:**
```python
# Analyze portfolio risk
response = await advisor.analyze_portfolio_risk(
    portfolio=current_positions,
    market_conditions={"vix": 25, "regime": "volatile"},
    scenarios=[
        {"name": "Market Crash", "spy_change": -0.20},
        {"name": "Rate Hike", "rate_change": 0.005}
    ]
)
```

### 3. Portfolio Strategist

**Expertise:**
- Modern Portfolio Theory and beyond
- Factor investing and smart beta
- Dynamic asset allocation
- Portfolio optimization techniques
- Behavioral finance considerations

**Portfolio Optimization:**
```python
# Optimize portfolio allocation
response = await advisor.optimize_portfolio_allocation(
    current_portfolio=positions,
    constraints={
        "max_position_size": 0.20,
        "min_positions": 5,
        "sector_limits": {"TECH": 0.40}
    },
    objectives=["maximize_sharpe", "minimize_drawdown"]
)
```

### 4. Market Researcher

**Expertise:**
- Market microstructure analysis
- Sentiment analysis and alternative data
- Macroeconomic indicators
- Cross-market relationships
- Emerging market trends

### 5. Trading Advisor

**Expertise:**
- High-frequency trading strategies
- Market making algorithms
- Order execution optimization
- Transaction cost analysis
- Trading signal generation

## API Integration

### Configuration Endpoint

**GET AI-Recommended Analysis Configuration**
```bash
POST /api/ai/analysis/config
{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "objectives": ["maximize_returns", "minimize_risk"],
    "constraints": {
        "timeframe": "1d",
        "analysis_type": "comprehensive",
        "capital": 100000
    }
}
```

**Response:**
```json
{
    "status": "success",
    "configuration": {
        "idtxl": {
            "enabled": true,
            "parameters": {
                "max_lag": 10,
                "cmi_estimator": "JidtGaussianCMI",
                "n_perm_max_stat": 500
            }
        },
        "ml": {
            "enabled": true,
            "models": ["random_forest", "xgboost"],
            "features": ["rsi", "macd", "volume_ratio"]
        },
        "risk_controls": {
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "var_limit": 0.05
        }
    }
}
```

### Results Interpretation

**POST Interpret Analysis Results**
```bash
POST /api/ai/analysis/interpret
{
    "results": {
        "transfer_entropy": {
            "AAPL->MSFT": 0.045,
            "MSFT->GOOGL": 0.032
        },
        "ml_predictions": {
            "accuracy": 0.68,
            "feature_importance": {
                "rsi": 0.35,
                "volume": 0.28
            }
        }
    },
    "context": {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "timeframe": "1d"
    }
}
```

### Interactive Consultation

**POST Chat with AI Advisor**
```bash
POST /api/ai/consult
{
    "question": "How should I adjust my portfolio given the recent inflation data?",
    "context": {
        "portfolio_beta": 1.2,
        "current_allocation": {
            "stocks": 0.70,
            "bonds": 0.20,
            "commodities": 0.10
        }
    }
}
```

## Frontend Integration

### AI Advisor Component

Located in `frontend/src/components/AIAdvisor.tsx`

**Features:**
- Real-time chat interface
- Analysis configuration assistant
- Strategy building guidance
- Risk consultation

**Usage:**
```typescript
import { AIAdvisor } from './components/AIAdvisor';

function App() {
    return (
        <AIAdvisor 
            initialRole="quantitative_analyst"
            context={{
                portfolio: currentPortfolio,
                marketConditions: latestMarketData
            }}
        />
    );
}
```

### Chat Interface

The chat interface provides:
- Natural language interaction
- Context-aware responses
- Code examples and formulas
- Visual explanations

### Analysis Assistant

Helps users configure:
- Optimal IDTxl parameters
- Feature selection for ML
- Neural network architectures
- Risk thresholds

## Advanced Features

### 1. Strategy Validation

```python
# Validate trading strategy logic
validation = await advisor.validate_strategy_logic(
    strategy_code="""
    if rsi < 30 and macd_histogram > 0:
        signal = 'BUY'
    elif rsi > 70 and macd_histogram < 0:
        signal = 'SELL'
    """,
    strategy_description="RSI-MACD momentum strategy"
)
```

**Validation Results:**
- Logical flaws detection
- Statistical validity assessment
- Risk management evaluation
- Performance estimation
- Edge case identification

### 2. Research Report Generation

```python
# Generate professional research report
report = await advisor.generate_research_report(
    topic="Transfer Entropy Analysis of Tech Sector",
    data={
        "findings": analysis_results,
        "methodology": "IDTxl with 10-day lag",
        "period": "2024-01-01 to 2024-12-31"
    },
    analysis_type="causality_study"
)
```

**Report Sections:**
- Executive Summary
- Methodology
- Key Findings
- Statistical Analysis
- Risk Considerations
- Investment Implications

### 3. Market Regime Analysis

```python
# Analyze current market regime
regime = await advisor.analyze_market_regime(
    market_data={
        "vix": 22.5,
        "yield_curve": {"2y": 4.5, "10y": 4.2},
        "sector_rotation": "defensive"
    },
    economic_indicators={
        "inflation": 3.2,
        "unemployment": 3.7,
        "gdp_growth": 2.1
    }
)
```

## Configuration

### Environment Variables

```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_claude_key  # Optional

# Model Selection
AI_PROVIDER=openai  # or anthropic
AI_MODEL=gpt-4-turbo-preview

# Response Settings
AI_MAX_TOKENS=2000
AI_TEMPERATURE=0.7
```

### Model Parameters

```python
# Configure AI advisor
ai_advisor = AIAdvisorService(
    provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4-turbo-preview",
    temperature=0.7,
    max_tokens=2000
)
```

## Best Practices

### 1. Context Management

**Provide Rich Context:**
```python
context = AnalysisContext(
    symbols=symbols,
    timeframe=timeframe,
    analysis_type=analysis_type,
    objectives=objectives,
    risk_tolerance=risk_tolerance,
    capital=capital,
    current_positions=positions,  # Include current holdings
    market_conditions=market_data,  # Current market state
    historical_performance=past_results  # Past performance
)
```

### 2. Response Validation

Always validate AI responses:
```python
def validate_ai_recommendation(response):
    # Check parameter ranges
    if response.recommendations:
        for rec in response.recommendations:
            if "stop_loss" in rec.text:
                # Extract and validate stop loss percentage
                stop_loss = extract_percentage(rec.text)
                assert 0.001 <= stop_loss <= 0.10
    
    # Verify confidence level
    assert 0 <= response.confidence_level <= 1.0
```

### 3. Error Handling

```python
try:
    response = await advisor.get_analysis_recommendations(context)
except RateLimitError:
    # Use cached recommendations
    response = get_cached_recommendations(context)
except APIError as e:
    # Fallback to rule-based recommendations
    response = get_fallback_recommendations(context)
```

## Use Cases

### 1. Pre-Trade Analysis

```python
# Before executing trades
advisor_check = await advisor.advisor.recommend_trading_strategy(
    analysis_results={
        "signals": ["BUY_AAPL", "SELL_MSFT"],
        "confidence": [0.75, 0.68]
    },
    market_data=current_market_data,
    risk_profile={
        "max_drawdown": 0.15,
        "position_limits": {"AAPL": 1000}
    }
)

if advisor_check.warnings:
    # Review warnings before proceeding
    for warning in advisor_check.warnings:
        log_warning(warning)
```

### 2. Post-Analysis Interpretation

```python
# After running IDTxl analysis
interpretation = await advisor.interpret_analysis_results(
    results=idtxl_results,
    context=analysis_context
)

# Extract actionable insights
for insight in interpretation.insights:
    if "significant causality" in insight:
        # Flag for trading strategy
        add_to_watchlist(extract_symbols(insight))
```

### 3. Continuous Learning

```python
# Regular strategy review
async def weekly_strategy_review():
    performance = calculate_weekly_performance()
    
    consultation = await advisor.interactive_consultation(
        user_id="system",
        question=f"Review this week's performance: {performance}",
        context={
            "trades_executed": weekly_trades,
            "market_conditions": weekly_market_summary
        }
    )
    
    # Apply learnings
    apply_strategy_adjustments(consultation)
```

## Security Considerations

### API Key Management

- Store API keys in secure vault
- Rotate keys regularly
- Monitor usage and costs
- Implement rate limiting

### Data Privacy

- Don't send sensitive user data
- Anonymize portfolio details
- Use aggregated metrics
- Implement data retention policies

### Response Filtering

```python
def filter_ai_response(response):
    # Remove any potential PII
    response = remove_pii(response)
    
    # Filter investment advice disclaimers
    response = add_disclaimers(response)
    
    # Validate no malicious content
    response = sanitize_output(response)
    
    return response
```

## Performance Optimization

### Caching Strategy

```python
# Cache common queries
@cache(ttl=3600)  # 1 hour cache
async def get_market_regime_analysis(market_data):
    return await advisor.analyze_market_regime(market_data)
```

### Batch Processing

```python
# Batch multiple consultations
questions = [
    "Optimal RSI period for AAPL?",
    "Best ML features for MSFT?",
    "Risk limits for GOOGL?"
]

responses = await advisor.batch_consultation(questions)
```

### Token Optimization

- Use concise prompts
- Request specific information
- Limit response length when appropriate
- Stream responses for real-time feedback

## Monitoring and Analytics

### Usage Metrics

Track:
- API calls per user
- Response times
- Token usage
- Error rates
- User satisfaction

### Quality Metrics

Monitor:
- Recommendation accuracy
- User follow-through rate
- Performance improvement
- Risk prevention success

### Cost Management

```python
# Monitor API costs
async def log_api_usage(user_id, tokens_used, cost):
    await db.insert({
        "user_id": user_id,
        "tokens": tokens_used,
        "cost": cost,
        "timestamp": datetime.now()
    })
    
    # Alert if exceeding budget
    if await get_monthly_cost(user_id) > BUDGET_LIMIT:
        await notify_admin(f"User {user_id} exceeding AI budget")
```

## Recent Improvements (August 2025)

### Dynamic AI-Powered Recommendations

The AI Advisory system has been significantly enhanced to provide true PhD-level expertise:

#### Before (Hardcoded Rules)
```javascript
// OLD: Static rules in AIAdvisor.tsx
if (idtxl.permutations < 500 && idtxl.significanceLevel < 0.01) {
  recommendations.push({
    text: 'For significance level < 0.01, increase permutations to at least 500',
    priority: 'high'
  });
}
```

#### After (Dynamic AI Analysis)
```javascript
// NEW: Actual AI analysis
const response = await api.post('/api/ai-advisor/contextual-analysis', {
  context: analysisContext,  // Full configuration details
  requestType: 'configuration_optimization'
});

// AI analyzes the specific configuration and returns tailored advice
setAdvisoryResponse({
  recommendations: response.data.data.recommendations,  // AI-generated
  insights: response.data.data.insights,              // Context-aware
  warnings: response.data.data.warnings,              // Specific to config
  optimal_configuration: response.data.data.optimal_configuration
});
```

### Key Improvements

1. **Full Context Awareness**
   - AI receives complete configuration parameters
   - Analyzes specific symbol combinations
   - Considers date ranges and data characteristics

2. **Real-Time Updates**
   - Configuration changes trigger immediate AI analysis
   - Debounced API calls for performance
   - Loading states for better UX

3. **Academic Rigor**
   - Recommendations based on statistical theory
   - Citations to research when applicable
   - Explanations of parameter interactions

4. **Optimal Configuration Suggestions**
   - AI extracts optimal values from its analysis
   - Considers computational vs. accuracy tradeoffs
   - Tailored to specific use cases

### Frontend Integration

The Analysis page now features:
- Real-time AI recommendations that update as you type
- Context-aware suggestions based on selected analysis type
- Loading indicators while AI processes
- Graceful error handling with fallbacks

### Backend Enhancements

- `AnalysisContext` now includes `additional_context` field
- Full configuration passed to AI service
- Enhanced prompt engineering for specific recommendations
- Parameter extraction from AI responses

## Future Enhancements

1. **Multi-Modal Analysis**: Incorporate chart images and audio
2. **Real-Time Collaboration**: AI-assisted team analysis
3. **Custom Model Fine-Tuning**: Domain-specific models
4. **Automated Report Generation**: Scheduled insights
5. **Voice Interface**: Natural language commands
6. **Caching Layer**: Intelligent caching of common configurations
7. **A/B Testing**: Compare AI vs rule-based recommendations