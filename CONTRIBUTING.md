# Contributing to Financial Time Series Analysis Platform

Thank you for your interest in contributing to the Financial Time Series Analysis Platform! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Coding Standards](#coding-standards)
5. [Testing Requirements](#testing-requirements)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Issue Reporting](#issue-reporting)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We pledge to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Expected Behavior

- Be professional and respectful in all interactions
- Provide constructive feedback
- Accept responsibility for mistakes
- Focus on resolution rather than blame
- Help maintain a positive environment

## Getting Started

### Prerequisites

1. **Development Environment**
   ```bash
   # Required tools
   - Python 3.11+
   - Node.js 18+
   - Docker & Docker Compose
   - Git
   ```

2. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub
   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/financial-platform.git
   cd financial-platform
   
   # Add upstream remote
   git remote add upstream https://github.com/ORIGINAL_OWNER/financial-platform.git
   ```

3. **Setup Development Environment**
   ```bash
   # Backend setup
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   
   # Frontend setup
   cd ../frontend
   npm install
   
   # Start services
   docker-compose -f docker-compose.dev.yml up -d
   ```

## Development Process

### Branch Strategy

We use Git Flow for our branching strategy:

```
main         - Production-ready code
develop      - Integration branch for features
feature/*    - New features
bugfix/*     - Bug fixes
hotfix/*     - Urgent production fixes
release/*    - Release preparation
```

### Creating a Feature Branch

```bash
# Update your local repository
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
git add .
git commit -m "feat: add new feature"

# Push to your fork
git push origin feature/your-feature-name
```

## Coding Standards

### Python (Backend)

#### Style Guide
We follow PEP 8 with the following additions:
- Line length: 88 characters (Black formatter)
- Use type hints for all functions
- Docstrings for all public functions (Google style)

#### Example
```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for running financial analysis.
    
    This service provides methods for running various types of
    financial analysis including IDTxl, ML, and neural networks.
    """
    
    async def run_analysis(
        self,
        symbols: List[str],
        analysis_type: str,
        parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run analysis on given symbols.
        
        Args:
            symbols: List of stock symbols to analyze
            analysis_type: Type of analysis ('idtxl', 'ml', 'nn')
            parameters: Optional analysis parameters
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        if not symbols:
            raise ValueError("Symbols list cannot be empty")
            
        parameters = parameters or {}
        logger.info(f"Running {analysis_type} analysis on {symbols}")
        
        # Implementation here
        return results
```

#### Linting and Formatting
```bash
# Format code
black backend/
isort backend/

# Lint code
flake8 backend/
mypy backend/
```

### TypeScript/React (Frontend)

#### Style Guide
- Use functional components with hooks
- TypeScript for all files
- ESLint and Prettier for formatting

#### Example
```typescript
import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { AnalysisResult } from '@/types';
import { analysisService } from '@/services';

interface AnalysisProps {
  symbols: string[];
  onComplete?: (result: AnalysisResult) => void;
}

export const Analysis: React.FC<AnalysisProps> = ({ symbols, onComplete }) => {
  const [isRunning, setIsRunning] = useState(false);
  
  const { data, error, isLoading } = useQuery({
    queryKey: ['analysis', symbols],
    queryFn: () => analysisService.runAnalysis(symbols),
    enabled: symbols.length > 0,
  });
  
  useEffect(() => {
    if (data && onComplete) {
      onComplete(data);
    }
  }, [data, onComplete]);
  
  if (error) {
    return <ErrorDisplay error={error} />;
  }
  
  return (
    <div className="analysis-container">
      {/* Component content */}
    </div>
  );
};
```

#### Linting and Formatting
```bash
# Format code
npm run prettier

# Lint code
npm run lint

# Type check
npm run typecheck
```

### Git Commit Messages

We follow the Conventional Commits specification:

```
feat: add new analysis method
fix: correct transfer entropy calculation
docs: update API documentation
style: format code with black
refactor: extract common analysis logic
test: add tests for risk management
chore: update dependencies
```

Format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

## Testing Requirements

### Backend Testing

#### Unit Tests
```python
# test_analysis_service.py
import pytest
from unittest.mock import Mock, patch

from app.services.analysis import AnalysisService


class TestAnalysisService:
    @pytest.fixture
    def service(self):
        return AnalysisService()
    
    @pytest.mark.asyncio
    async def test_run_analysis_success(self, service):
        # Arrange
        symbols = ["AAPL", "MSFT"]
        analysis_type = "idtxl"
        
        # Act
        result = await service.run_analysis(symbols, analysis_type)
        
        # Assert
        assert result["status"] == "completed"
        assert "transfer_entropy" in result
    
    @pytest.mark.asyncio
    async def test_run_analysis_empty_symbols(self, service):
        # Arrange
        symbols = []
        
        # Act & Assert
        with pytest.raises(ValueError, match="Symbols list cannot be empty"):
            await service.run_analysis(symbols, "idtxl")
```

#### Integration Tests
```python
# test_integration_trading.py
import pytest
from httpx import AsyncClient

from main import app


@pytest.mark.asyncio
async def test_complete_trading_flow():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Login
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "test", "password": "test"}
        )
        token = login_response.json()["access_token"]
        
        # Place order
        headers = {"Authorization": f"Bearer {token}"}
        order_response = await client.post(
            "/api/trading/orders",
            json={
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "order_type": "market"
            },
            headers=headers
        )
        
        assert order_response.status_code == 201
        assert order_response.json()["status"] == "pending"
```

### Frontend Testing

#### Component Tests
```typescript
// Analysis.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Analysis } from './Analysis';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false } },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

describe('Analysis Component', () => {
  it('should render analysis form', () => {
    render(<Analysis symbols={['AAPL']} />, { wrapper });
    
    expect(screen.getByText('Run Analysis')).toBeInTheDocument();
    expect(screen.getByLabelText('Analysis Type')).toBeInTheDocument();
  });
  
  it('should handle analysis submission', async () => {
    const onComplete = jest.fn();
    render(
      <Analysis symbols={['AAPL']} onComplete={onComplete} />,
      { wrapper }
    );
    
    const submitButton = screen.getByText('Run Analysis');
    await userEvent.click(submitButton);
    
    await waitFor(() => {
      expect(onComplete).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'completed',
        })
      );
    });
  });
});
```

### Test Coverage Requirements

- Minimum 80% code coverage
- All critical paths must be tested
- Integration tests for all API endpoints
- E2E tests for critical user flows

## Documentation

### Code Documentation

#### Python Docstrings
```python
def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """Calculate the Sharpe ratio of a returns series.
    
    The Sharpe ratio is the average return earned in excess of the
    risk-free rate per unit of volatility.
    
    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate (default: 0.02)
        periods_per_year: Number of periods in a year (default: 252)
        
    Returns:
        Sharpe ratio
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")
        Sharpe Ratio: 1.25
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
```

#### TypeScript JSDoc
```typescript
/**
 * Calculates technical indicators for the given price data
 * 
 * @param prices - Array of price data points
 * @param indicators - List of indicators to calculate
 * @returns Object containing calculated indicator values
 * 
 * @example
 * const indicators = calculateIndicators(prices, ['rsi', 'macd']);
 * console.log(indicators.rsi); // [45.2, 48.1, 52.3, ...]
 */
export function calculateIndicators(
  prices: PriceData[],
  indicators: IndicatorType[]
): IndicatorResults {
  // Implementation
}
```

### API Documentation

All API endpoints must be documented with:
- Description
- Parameters
- Request/Response examples
- Error codes
- Rate limits

### User Documentation

- Update user guides for new features
- Include screenshots where helpful
- Provide code examples
- Update FAQ section

## Pull Request Process

### Before Submitting

1. **Update your branch**
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout feature/your-feature
   git rebase develop
   ```

2. **Run all tests**
   ```bash
   # Backend
   cd backend
   pytest
   
   # Frontend
   cd frontend
   npm test
   ```

3. **Check code quality**
   ```bash
   # Backend
   black backend/
   flake8 backend/
   mypy backend/
   
   # Frontend
   npm run lint
   npm run typecheck
   ```

4. **Update documentation**
   - Add/update docstrings
   - Update README if needed
   - Add to CHANGELOG.md

### Creating the Pull Request

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature
   ```

2. **Create PR on GitHub**
   - Use a descriptive title
   - Reference related issues
   - Provide detailed description
   - Include screenshots if UI changes

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No new warnings
   
   ## Related Issues
   Fixes #123
   ```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline must pass
   - Code coverage maintained
   - No security vulnerabilities

2. **Code Review**
   - At least 2 approvals required
   - Address all feedback
   - Resolve all conversations

3. **Merge**
   - Squash and merge for features
   - Rebase and merge for fixes
   - Update CHANGELOG.md

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What should happen

**Screenshots**
If applicable

**Environment**
- OS: [e.g. Windows 10]
- Browser: [e.g. Chrome 96]
- Version: [e.g. 1.0.0]

**Additional context**
Any other relevant information
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution**
What you want to happen

**Alternatives considered**
Other solutions you've considered

**Additional context**
Any other information
```

## Questions?

- Check existing issues and discussions
- Join our Discord server
- Email: dev-support@finplatform.com

Thank you for contributing to the Financial Time Series Analysis Platform!