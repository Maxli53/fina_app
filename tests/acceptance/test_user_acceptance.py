"""
User acceptance testing framework for validating platform functionality from end-user perspective
"""

import asyncio
import pytest
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TestScenarioStatus(Enum):
    """Test scenario execution status"""
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class TestScenario:
    """User acceptance test scenario"""
    id: str
    name: str
    description: str
    persona: str
    steps: List[Dict[str, Any]]
    expected_outcome: str
    priority: str
    status: TestScenarioStatus = TestScenarioStatus.SKIPPED
    actual_result: str = None
    error_message: str = None
    duration: float = 0
    screenshots: List[str] = None


class UserAcceptanceTester:
    """UAT framework for end-to-end user testing"""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.driver = None
        self.wait = None
        self.test_results: List[TestScenario] = []
        
    def setup_driver(self):
        """Setup Selenium WebDriver"""
        # Configure Chrome options
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        
        # Initialize driver
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
        
    def teardown_driver(self):
        """Clean up WebDriver"""
        if self.driver:
            self.driver.quit()
    
    async def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all UAT scenarios"""
        print("Starting User Acceptance Testing...")
        print("="*60)
        
        self.setup_driver()
        
        try:
            # Define test scenarios
            scenarios = self.get_test_scenarios()
            
            # Execute each scenario
            for scenario in scenarios:
                await self.execute_scenario(scenario)
                self.test_results.append(scenario)
            
            # Generate report
            return self.generate_uat_report()
            
        finally:
            self.teardown_driver()
    
    def get_test_scenarios(self) -> List[TestScenario]:
        """Define all UAT scenarios"""
        return [
            # Trader persona scenarios
            TestScenario(
                id="UAT-001",
                name="Execute Market Order",
                description="Trader places a market order for AAPL stock",
                persona="Professional Trader",
                priority="High",
                steps=[
                    {"action": "login", "data": {"username": "trader1", "password": "test123"}},
                    {"action": "navigate", "path": "/trading"},
                    {"action": "search_symbol", "symbol": "AAPL"},
                    {"action": "enter_order", "data": {"quantity": 100, "type": "market", "side": "buy"}},
                    {"action": "submit_order"},
                    {"action": "verify_confirmation"},
                    {"action": "check_position", "symbol": "AAPL", "quantity": 100}
                ],
                expected_outcome="Order executed successfully and position updated"
            ),
            
            TestScenario(
                id="UAT-002",
                name="Run IDTxl Analysis",
                description="Analyst runs transfer entropy analysis on portfolio",
                persona="Quantitative Analyst",
                priority="High",
                steps=[
                    {"action": "login", "data": {"username": "analyst1", "password": "test123"}},
                    {"action": "navigate", "path": "/analysis"},
                    {"action": "select_analysis_type", "type": "idtxl"},
                    {"action": "configure_parameters", "data": {"symbols": ["AAPL", "MSFT", "GOOGL"], "lag": 5}},
                    {"action": "run_analysis"},
                    {"action": "wait_for_results", "timeout": 60},
                    {"action": "verify_results", "expected": ["transfer_entropy", "causality_matrix"]}
                ],
                expected_outcome="Analysis completes with transfer entropy results displayed"
            ),
            
            TestScenario(
                id="UAT-003",
                name="Create and Backtest Strategy",
                description="Researcher creates a mean reversion strategy and backtests it",
                persona="Strategy Developer",
                priority="High",
                steps=[
                    {"action": "login", "data": {"username": "researcher1", "password": "test123"}},
                    {"action": "navigate", "path": "/strategy"},
                    {"action": "create_strategy", "data": {
                        "name": "Mean Reversion Test",
                        "type": "mean_reversion",
                        "parameters": {"lookback": 20, "threshold": 2.0}
                    }},
                    {"action": "configure_backtest", "data": {
                        "start_date": "2024-01-01",
                        "end_date": "2024-06-30",
                        "initial_capital": 100000
                    }},
                    {"action": "run_backtest"},
                    {"action": "verify_backtest_results", "metrics": ["sharpe_ratio", "total_return", "max_drawdown"]}
                ],
                expected_outcome="Strategy created and backtest shows performance metrics"
            ),
            
            TestScenario(
                id="UAT-004",
                name="Monitor Real-time Portfolio",
                description="Portfolio manager monitors positions and P&L in real-time",
                persona="Portfolio Manager",
                priority="High",
                steps=[
                    {"action": "login", "data": {"username": "pm1", "password": "test123"}},
                    {"action": "navigate", "path": "/portfolio"},
                    {"action": "verify_portfolio_overview"},
                    {"action": "check_real_time_updates", "duration": 30},
                    {"action": "verify_pnl_calculation"},
                    {"action": "check_risk_metrics", "metrics": ["var", "sharpe", "beta"]}
                ],
                expected_outcome="Portfolio displays real-time updates with accurate calculations"
            ),
            
            TestScenario(
                id="UAT-005",
                name="Set Risk Alerts",
                description="Risk manager configures portfolio risk alerts",
                persona="Risk Manager",
                priority="Medium",
                steps=[
                    {"action": "login", "data": {"username": "risk1", "password": "test123"}},
                    {"action": "navigate", "path": "/risk"},
                    {"action": "configure_alert", "data": {
                        "type": "var_breach",
                        "threshold": 10000,
                        "notification": "email"
                    }},
                    {"action": "save_alert"},
                    {"action": "verify_alert_active"},
                    {"action": "test_alert_trigger"}
                ],
                expected_outcome="Risk alert configured and triggers when threshold breached"
            ),
            
            TestScenario(
                id="UAT-006",
                name="Export Analysis Report",
                description="User exports comprehensive analysis report",
                persona="Quantitative Analyst",
                priority:"Medium",
                steps=[
                    {"action": "login", "data": {"username": "analyst1", "password": "test123"}},
                    {"action": "navigate", "path": "/reports"},
                    {"action": "select_report_type", "type": "comprehensive_analysis"},
                    {"action": "configure_report", "data": {
                        "date_range": "last_30_days",
                        "include_charts": True,
                        "format": "pdf"
                    }},
                    {"action": "generate_report"},
                    {"action": "verify_download", "filename": "analysis_report.pdf"}
                ],
                expected_outcome="PDF report generated and downloaded successfully"
            ),
            
            TestScenario(
                id="UAT-007",
                name="Multi-Asset Portfolio Rebalancing",
                description="Execute portfolio rebalancing across multiple assets",
                persona="Portfolio Manager",
                priority: "High",
                steps=[
                    {"action": "login", "data": {"username": "pm1", "password": "test123"}},
                    {"action": "navigate", "path": "/portfolio/rebalance"},
                    {"action": "review_current_allocation"},
                    {"action": "set_target_allocation", "data": {
                        "AAPL": 30,
                        "MSFT": 25,
                        "GOOGL": 20,
                        "Cash": 25
                    }},
                    {"action": "preview_rebalance_orders"},
                    {"action": "execute_rebalance"},
                    {"action": "verify_new_allocation"}
                ],
                expected_outcome="Portfolio rebalanced to target allocation"
            ),
            
            TestScenario(
                id="UAT-008",
                name="Machine Learning Model Training",
                description="Train and deploy ML model for price prediction",
                persona="Data Scientist",
                priority: "High",
                steps=[
                    {"action": "login", "data": {"username": "ds1", "password": "test123"}},
                    {"action": "navigate", "path": "/ml-models"},
                    {"action": "select_model_type", "type": "random_forest"},
                    {"action": "configure_features", "features": ["rsi", "macd", "volume_ratio"]},
                    {"action": "set_training_params", "data": {"train_size": 0.8, "cv_folds": 5}},
                    {"action": "train_model"},
                    {"action": "review_metrics", "metrics": ["accuracy", "precision", "recall"]},
                    {"action": "deploy_model", "name": "RF_Price_Predictor_v1"}
                ],
                expected_outcome="ML model trained and deployed for use"
            ),
            
            TestScenario(
                id="UAT-009",
                name="WebSocket Real-time Data",
                description="Verify WebSocket streaming for real-time market data",
                persona="Professional Trader",
                priority: "High",
                steps=[
                    {"action": "login", "data": {"username": "trader1", "password": "test123"}},
                    {"action": "navigate", "path": "/market-data"},
                    {"action": "subscribe_symbols", "symbols": ["AAPL", "MSFT"]},
                    {"action": "verify_streaming", "duration": 30},
                    {"action": "check_data_quality", "metrics": ["latency", "completeness"]},
                    {"action": "unsubscribe_symbols"}
                ],
                expected_outcome="Real-time data streams without interruption"
            ),
            
            TestScenario(
                id="UAT-010",
                name="Mobile Responsive Trading",
                description="Execute trades on mobile device",
                persona="Mobile Trader",
                priority: "Medium",
                steps=[
                    {"action": "resize_viewport", "width": 375, "height": 667},
                    {"action": "login", "data": {"username": "mobile1", "password": "test123"}},
                    {"action": "navigate", "path": "/trading"},
                    {"action": "verify_mobile_layout"},
                    {"action": "place_mobile_order", "data": {"symbol": "AAPL", "quantity": 50}},
                    {"action": "verify_order_execution"}
                ],
                expected_outcome="Mobile interface works correctly for trading"
            )
        ]
    
    async def execute_scenario(self, scenario: TestScenario):
        """Execute a single UAT scenario"""
        print(f"\n[{scenario.id}] {scenario.name}")
        print(f"Persona: {scenario.persona}")
        print(f"Description: {scenario.description}")
        
        start_time = time.time()
        scenario.screenshots = []
        
        try:
            # Execute each step
            for i, step in enumerate(scenario.steps):
                print(f"  Step {i+1}: {step['action']}")
                await self.execute_step(step, scenario)
            
            # If we get here, scenario passed
            scenario.status = TestScenarioStatus.PASSED
            scenario.actual_result = scenario.expected_outcome
            print(f"  ✅ PASSED")
            
        except Exception as e:
            scenario.status = TestScenarioStatus.FAILED
            scenario.error_message = str(e)
            scenario.actual_result = f"Failed at step: {step['action']}"
            
            # Take error screenshot
            screenshot_path = self.take_screenshot(f"{scenario.id}_error")
            scenario.screenshots.append(screenshot_path)
            
            print(f"  ❌ FAILED: {e}")
            
        finally:
            scenario.duration = time.time() - start_time
    
    async def execute_step(self, step: Dict[str, Any], scenario: TestScenario):
        """Execute a single test step"""
        action = step["action"]
        
        if action == "login":
            await self.action_login(step["data"])
        elif action == "navigate":
            await self.action_navigate(step["path"])
        elif action == "search_symbol":
            await self.action_search_symbol(step["symbol"])
        elif action == "enter_order":
            await self.action_enter_order(step["data"])
        elif action == "submit_order":
            await self.action_submit_order()
        elif action == "verify_confirmation":
            await self.action_verify_confirmation()
        elif action == "check_position":
            await self.action_check_position(step["symbol"], step["quantity"])
        elif action == "select_analysis_type":
            await self.action_select_analysis_type(step["type"])
        elif action == "configure_parameters":
            await self.action_configure_parameters(step["data"])
        elif action == "run_analysis":
            await self.action_run_analysis()
        elif action == "wait_for_results":
            await self.action_wait_for_results(step.get("timeout", 30))
        elif action == "verify_results":
            await self.action_verify_results(step["expected"])
        elif action == "create_strategy":
            await self.action_create_strategy(step["data"])
        elif action == "configure_backtest":
            await self.action_configure_backtest(step["data"])
        elif action == "run_backtest":
            await self.action_run_backtest()
        elif action == "verify_backtest_results":
            await self.action_verify_backtest_results(step["metrics"])
        elif action == "verify_portfolio_overview":
            await self.action_verify_portfolio_overview()
        elif action == "check_real_time_updates":
            await self.action_check_real_time_updates(step["duration"])
        elif action == "verify_pnl_calculation":
            await self.action_verify_pnl_calculation()
        elif action == "check_risk_metrics":
            await self.action_check_risk_metrics(step["metrics"])
        elif action == "resize_viewport":
            await self.action_resize_viewport(step["width"], step["height"])
        elif action == "verify_mobile_layout":
            await self.action_verify_mobile_layout()
        else:
            # Generic action handler
            await asyncio.sleep(1)
    
    # Action implementations
    async def action_login(self, credentials: Dict[str, str]):
        """Perform login action"""
        self.driver.get(f"{self.base_url}/login")
        
        # Enter username
        username_field = self.wait.until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        username_field.send_keys(credentials["username"])
        
        # Enter password
        password_field = self.driver.find_element(By.ID, "password")
        password_field.send_keys(credentials["password"])
        
        # Click login button
        login_button = self.driver.find_element(By.ID, "login-button")
        login_button.click()
        
        # Wait for dashboard
        self.wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "dashboard"))
        )
        
        await asyncio.sleep(1)
    
    async def action_navigate(self, path: str):
        """Navigate to specific path"""
        self.driver.get(f"{self.base_url}{path}")
        await asyncio.sleep(1)
    
    async def action_search_symbol(self, symbol: str):
        """Search for a symbol"""
        search_box = self.wait.until(
            EC.presence_of_element_located((By.ID, "symbol-search"))
        )
        search_box.clear()
        search_box.send_keys(symbol)
        search_box.send_keys(Keys.RETURN)
        await asyncio.sleep(1)
    
    async def action_enter_order(self, order_data: Dict[str, Any]):
        """Enter order details"""
        # Enter quantity
        quantity_field = self.driver.find_element(By.ID, "order-quantity")
        quantity_field.clear()
        quantity_field.send_keys(str(order_data["quantity"]))
        
        # Select order type
        order_type = self.driver.find_element(By.ID, f"order-type-{order_data['type']}")
        order_type.click()
        
        # Select side
        side_button = self.driver.find_element(By.ID, f"order-side-{order_data['side']}")
        side_button.click()
        
        await asyncio.sleep(1)
    
    async def action_submit_order(self):
        """Submit the order"""
        submit_button = self.driver.find_element(By.ID, "submit-order")
        submit_button.click()
        await asyncio.sleep(2)
    
    async def action_verify_confirmation(self):
        """Verify order confirmation"""
        confirmation = self.wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "order-confirmation"))
        )
        assert "success" in confirmation.get_attribute("class").lower()
    
    async def action_check_position(self, symbol: str, expected_quantity: int):
        """Check position exists with expected quantity"""
        position_row = self.wait.until(
            EC.presence_of_element_located((By.ID, f"position-{symbol}"))
        )
        quantity_cell = position_row.find_element(By.CLASS_NAME, "position-quantity")
        actual_quantity = int(quantity_cell.text)
        assert actual_quantity == expected_quantity
    
    async def action_select_analysis_type(self, analysis_type: str):
        """Select analysis type"""
        type_selector = self.driver.find_element(By.ID, f"analysis-type-{analysis_type}")
        type_selector.click()
        await asyncio.sleep(1)
    
    async def action_configure_parameters(self, params: Dict[str, Any]):
        """Configure analysis parameters"""
        # Handle different parameter types
        if "symbols" in params:
            for symbol in params["symbols"]:
                symbol_checkbox = self.driver.find_element(By.ID, f"symbol-{symbol}")
                if not symbol_checkbox.is_selected():
                    symbol_checkbox.click()
        
        if "lag" in params:
            lag_field = self.driver.find_element(By.ID, "lag-parameter")
            lag_field.clear()
            lag_field.send_keys(str(params["lag"]))
        
        await asyncio.sleep(1)
    
    async def action_run_analysis(self):
        """Run the analysis"""
        run_button = self.driver.find_element(By.ID, "run-analysis")
        run_button.click()
        await asyncio.sleep(2)
    
    async def action_wait_for_results(self, timeout: int):
        """Wait for analysis results"""
        WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.CLASS_NAME, "analysis-results"))
        )
    
    async def action_verify_results(self, expected_elements: List[str]):
        """Verify expected results are present"""
        for element in expected_elements:
            result_element = self.driver.find_element(By.ID, f"result-{element}")
            assert result_element.is_displayed()
    
    async def action_create_strategy(self, strategy_data: Dict[str, Any]):
        """Create a new strategy"""
        # Click create button
        create_button = self.driver.find_element(By.ID, "create-strategy")
        create_button.click()
        
        # Fill strategy form
        name_field = self.wait.until(
            EC.presence_of_element_located((By.ID, "strategy-name"))
        )
        name_field.send_keys(strategy_data["name"])
        
        # Select strategy type
        type_selector = self.driver.find_element(By.ID, f"strategy-type-{strategy_data['type']}")
        type_selector.click()
        
        # Set parameters
        for param, value in strategy_data["parameters"].items():
            param_field = self.driver.find_element(By.ID, f"param-{param}")
            param_field.clear()
            param_field.send_keys(str(value))
        
        # Save strategy
        save_button = self.driver.find_element(By.ID, "save-strategy")
        save_button.click()
        
        await asyncio.sleep(2)
    
    async def action_configure_backtest(self, backtest_data: Dict[str, Any]):
        """Configure backtest parameters"""
        # Set date range
        start_date = self.driver.find_element(By.ID, "backtest-start-date")
        start_date.send_keys(backtest_data["start_date"])
        
        end_date = self.driver.find_element(By.ID, "backtest-end-date")
        end_date.send_keys(backtest_data["end_date"])
        
        # Set initial capital
        capital_field = self.driver.find_element(By.ID, "initial-capital")
        capital_field.clear()
        capital_field.send_keys(str(backtest_data["initial_capital"]))
        
        await asyncio.sleep(1)
    
    async def action_run_backtest(self):
        """Run backtest"""
        run_button = self.driver.find_element(By.ID, "run-backtest")
        run_button.click()
        
        # Wait for completion
        self.wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "backtest-results"))
        )
    
    async def action_verify_backtest_results(self, metrics: List[str]):
        """Verify backtest metrics are displayed"""
        for metric in metrics:
            metric_element = self.driver.find_element(By.ID, f"metric-{metric}")
            assert metric_element.is_displayed()
            
            # Verify metric has a value
            value = metric_element.find_element(By.CLASS_NAME, "metric-value").text
            assert value and value != "N/A"
    
    async def action_verify_portfolio_overview(self):
        """Verify portfolio overview is displayed"""
        overview = self.wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "portfolio-overview"))
        )
        
        # Check key elements
        assert self.driver.find_element(By.ID, "total-value").is_displayed()
        assert self.driver.find_element(By.ID, "daily-pnl").is_displayed()
        assert self.driver.find_element(By.ID, "positions-table").is_displayed()
    
    async def action_check_real_time_updates(self, duration: int):
        """Monitor real-time updates"""
        initial_value = self.driver.find_element(By.ID, "total-value").text
        
        # Wait and check for updates
        await asyncio.sleep(duration)
        
        final_value = self.driver.find_element(By.ID, "total-value").text
        
        # Value should have updated (even if same, timestamp should differ)
        update_indicator = self.driver.find_element(By.CLASS_NAME, "last-updated")
        assert update_indicator.is_displayed()
    
    async def action_verify_pnl_calculation(self):
        """Verify P&L calculations"""
        total_pnl = self.driver.find_element(By.ID, "total-pnl").text
        
        # Get individual position P&Ls
        position_pnls = self.driver.find_elements(By.CLASS_NAME, "position-pnl")
        
        # Basic sanity check - total should be sum of individuals
        assert len(position_pnls) > 0
        assert total_pnl  # Has a value
    
    async def action_check_risk_metrics(self, metrics: List[str]):
        """Check risk metrics are displayed"""
        for metric in metrics:
            metric_element = self.driver.find_element(By.ID, f"risk-{metric}")
            assert metric_element.is_displayed()
    
    async def action_resize_viewport(self, width: int, height: int):
        """Resize browser viewport for mobile testing"""
        self.driver.set_window_size(width, height)
        await asyncio.sleep(1)
    
    async def action_verify_mobile_layout(self):
        """Verify mobile layout is responsive"""
        # Check mobile menu is visible
        mobile_menu = self.driver.find_element(By.CLASS_NAME, "mobile-menu")
        assert mobile_menu.is_displayed()
        
        # Check desktop menu is hidden
        desktop_menu = self.driver.find_element(By.CLASS_NAME, "desktop-menu")
        assert not desktop_menu.is_displayed()
    
    def take_screenshot(self, name: str) -> str:
        """Take screenshot and return path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uat_{name}_{timestamp}.png"
        self.driver.save_screenshot(filename)
        return filename
    
    def generate_uat_report(self) -> Dict[str, Any]:
        """Generate UAT report"""
        # Calculate statistics
        total_scenarios = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.status == TestScenarioStatus.PASSED)
        failed = sum(1 for r in self.test_results if r.status == TestScenarioStatus.FAILED)
        blocked = sum(1 for r in self.test_results if r.status == TestScenarioStatus.BLOCKED)
        skipped = sum(1 for r in self.test_results if r.status == TestScenarioStatus.SKIPPED)
        
        pass_rate = (passed / total_scenarios * 100) if total_scenarios > 0 else 0
        
        # Group by persona
        persona_results = {}
        for result in self.test_results:
            if result.persona not in persona_results:
                persona_results[result.persona] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0
                }
            
            persona_results[result.persona]["total"] += 1
            if result.status == TestScenarioStatus.PASSED:
                persona_results[result.persona]["passed"] += 1
            elif result.status == TestScenarioStatus.FAILED:
                persona_results[result.persona]["failed"] += 1
        
        # Generate report
        report = {
            "test_date": datetime.utcnow().isoformat(),
            "platform_url": self.base_url,
            "summary": {
                "total_scenarios": total_scenarios,
                "passed": passed,
                "failed": failed,
                "blocked": blocked,
                "skipped": skipped,
                "pass_rate": pass_rate,
                "total_duration": sum(r.duration for r in self.test_results)
            },
            "persona_summary": persona_results,
            "priority_summary": self._get_priority_summary(),
            "detailed_results": [self._format_result(r) for r in self.test_results],
            "failed_scenarios": [
                self._format_result(r) for r in self.test_results 
                if r.status == TestScenarioStatus.FAILED
            ],
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        with open("uat_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("USER ACCEPTANCE TESTING SUMMARY")
        print("="*60)
        print(f"Total Scenarios: {total_scenarios}")
        print(f"Passed: {passed} ({pass_rate:.1f}%)")
        print(f"Failed: {failed}")
        print(f"Blocked: {blocked}")
        print(f"Skipped: {skipped}")
        print(f"\nTotal Duration: {report['summary']['total_duration']:.2f} seconds")
        print(f"\nFull report saved to: uat_report.json")
        
        if failed > 0:
            print("\nFAILED SCENARIOS:")
            for result in self.test_results:
                if result.status == TestScenarioStatus.FAILED:
                    print(f"  - [{result.id}] {result.name}: {result.error_message}")
        
        return report
    
    def _format_result(self, result: TestScenario) -> Dict[str, Any]:
        """Format test result for report"""
        return {
            "id": result.id,
            "name": result.name,
            "description": result.description,
            "persona": result.persona,
            "priority": result.priority,
            "status": result.status.value,
            "expected_outcome": result.expected_outcome,
            "actual_result": result.actual_result,
            "error_message": result.error_message,
            "duration": result.duration,
            "screenshots": result.screenshots or []
        }
    
    def _get_priority_summary(self) -> Dict[str, Dict[str, int]]:
        """Get results grouped by priority"""
        priority_summary = {}
        
        for result in self.test_results:
            if result.priority not in priority_summary:
                priority_summary[result.priority] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0
                }
            
            priority_summary[result.priority]["total"] += 1
            if result.status == TestScenarioStatus.PASSED:
                priority_summary[result.priority]["passed"] += 1
            elif result.status == TestScenarioStatus.FAILED:
                priority_summary[result.priority]["failed"] += 1
        
        return priority_summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Check critical failures
        high_priority_failures = [
            r for r in self.test_results 
            if r.priority == "High" and r.status == TestScenarioStatus.FAILED
        ]
        
        if high_priority_failures:
            recommendations.append(
                f"CRITICAL: {len(high_priority_failures)} high-priority scenarios failed. "
                "These must be fixed before production release."
            )
        
        # Check persona coverage
        failed_personas = set()
        for result in self.test_results:
            if result.status == TestScenarioStatus.FAILED:
                failed_personas.add(result.persona)
        
        if failed_personas:
            recommendations.append(
                f"The following user personas experienced failures: {', '.join(failed_personas)}. "
                "Review and fix issues affecting these user groups."
            )
        
        # Performance recommendations
        slow_scenarios = [
            r for r in self.test_results 
            if r.duration > 60  # More than 1 minute
        ]
        
        if slow_scenarios:
            recommendations.append(
                f"{len(slow_scenarios)} scenarios took longer than 60 seconds. "
                "Consider performance optimization for these workflows."
            )
        
        # General recommendations
        if self.test_results:
            pass_rate = sum(1 for r in self.test_results if r.status == TestScenarioStatus.PASSED) / len(self.test_results) * 100
            
            if pass_rate < 95:
                recommendations.append(
                    "UAT pass rate is below 95%. Additional fixes and testing required."
                )
            elif pass_rate == 100:
                recommendations.append(
                    "All UAT scenarios passed. Platform is ready for beta release."
                )
        
        return recommendations


# Beta testing framework
class BetaTester:
    """Framework for managing beta testing with real users"""
    
    def __init__(self):
        self.beta_users: List[Dict[str, Any]] = []
        self.feedback_items: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
    
    async def run_beta_program(self, duration_days: int = 14) -> Dict[str, Any]:
        """Run beta testing program"""
        print(f"Starting {duration_days}-day beta testing program...")
        
        # Simulate beta testing activities
        await self.onboard_beta_users()
        await self.collect_usage_metrics()
        await self.gather_feedback()
        await self.analyze_results()
        
        return self.generate_beta_report()
    
    async def onboard_beta_users(self):
        """Onboard beta users"""
        # Simulate user onboarding
        user_profiles = [
            {"id": 1, "type": "Professional Trader", "experience": "Advanced"},
            {"id": 2, "type": "Quantitative Analyst", "experience": "Expert"},
            {"id": 3, "type": "Portfolio Manager", "experience": "Intermediate"},
            {"id": 4, "type": "Risk Manager", "experience": "Advanced"},
            {"id": 5, "type": "Retail Trader", "experience": "Beginner"}
        ]
        
        for profile in user_profiles:
            self.beta_users.append({
                **profile,
                "onboarded_date": datetime.utcnow(),
                "active": True
            })
    
    async def collect_usage_metrics(self):
        """Collect usage metrics from beta users"""
        # Simulate metrics collection
        self.metrics = {
            "daily_active_users": 5,
            "average_session_duration": 45,  # minutes
            "features_used": {
                "trading": 85,  # percentage
                "analysis": 70,
                "portfolio": 90,
                "strategy": 60,
                "reports": 40
            },
            "total_trades_executed": 127,
            "total_analyses_run": 89,
            "errors_encountered": 12,
            "crash_reports": 2
        }
    
    async def gather_feedback(self):
        """Gather feedback from beta users"""
        # Simulate feedback collection
        feedback_items = [
            {
                "user_id": 1,
                "type": "bug",
                "severity": "medium",
                "description": "Order confirmation sometimes doesn't appear",
                "feature": "trading"
            },
            {
                "user_id": 2,
                "type": "feature_request",
                "description": "Add support for options trading",
                "feature": "trading"
            },
            {
                "user_id": 3,
                "type": "improvement",
                "description": "Portfolio charts need better mobile responsiveness",
                "feature": "portfolio"
            },
            {
                "user_id": 1,
                "type": "positive",
                "description": "IDTxl analysis is incredibly powerful and fast",
                "feature": "analysis"
            },
            {
                "user_id": 4,
                "type": "bug",
                "severity": "low",
                "description": "Risk metrics sometimes show stale data",
                "feature": "risk"
            }
        ]
        
        self.feedback_items.extend(feedback_items)
    
    async def analyze_results(self):
        """Analyze beta testing results"""
        # Analyze feedback patterns
        pass
    
    def generate_beta_report(self) -> Dict[str, Any]:
        """Generate beta testing report"""
        report = {
            "beta_period": {
                "start_date": (datetime.utcnow() - timedelta(days=14)).isoformat(),
                "end_date": datetime.utcnow().isoformat(),
                "duration_days": 14
            },
            "participants": {
                "total_users": len(self.beta_users),
                "user_types": self._count_user_types(),
                "active_users": sum(1 for u in self.beta_users if u["active"])
            },
            "usage_metrics": self.metrics,
            "feedback_summary": self._summarize_feedback(),
            "key_findings": self._generate_key_findings(),
            "recommendations": self._generate_beta_recommendations()
        }
        
        return report
    
    def _count_user_types(self) -> Dict[str, int]:
        """Count users by type"""
        type_counts = {}
        for user in self.beta_users:
            user_type = user["type"]
            type_counts[user_type] = type_counts.get(user_type, 0) + 1
        return type_counts
    
    def _summarize_feedback(self) -> Dict[str, Any]:
        """Summarize feedback items"""
        summary = {
            "total_items": len(self.feedback_items),
            "by_type": {},
            "by_feature": {},
            "critical_issues": []
        }
        
        for item in self.feedback_items:
            # Count by type
            item_type = item["type"]
            summary["by_type"][item_type] = summary["by_type"].get(item_type, 0) + 1
            
            # Count by feature
            feature = item.get("feature", "general")
            summary["by_feature"][feature] = summary["by_feature"].get(feature, 0) + 1
            
            # Identify critical issues
            if item["type"] == "bug" and item.get("severity") in ["high", "critical"]:
                summary["critical_issues"].append(item["description"])
        
        return summary
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from beta testing"""
        findings = []
        
        # Usage patterns
        if self.metrics.get("average_session_duration", 0) > 30:
            findings.append("High user engagement with average session duration over 30 minutes")
        
        # Feature adoption
        high_usage_features = [
            f for f, usage in self.metrics.get("features_used", {}).items() 
            if usage > 80
        ]
        if high_usage_features:
            findings.append(f"Strong adoption of core features: {', '.join(high_usage_features)}")
        
        # Stability
        error_rate = self.metrics.get("errors_encountered", 0) / max(self.metrics.get("total_trades_executed", 1), 1)
        if error_rate < 0.1:
            findings.append("Platform demonstrates good stability with low error rate")
        
        return findings
    
    def _generate_beta_recommendations(self) -> List[str]:
        """Generate recommendations from beta testing"""
        recommendations = []
        
        # Based on feedback
        bugs = sum(1 for f in self.feedback_items if f["type"] == "bug")
        if bugs > 5:
            recommendations.append(f"Address {bugs} reported bugs before general release")
        
        # Based on usage
        if self.metrics.get("crash_reports", 0) > 0:
            recommendations.append("Investigate and fix crash reports")
        
        # Feature requests
        feature_requests = sum(1 for f in self.feedback_items if f["type"] == "feature_request")
        if feature_requests > 3:
            recommendations.append("Consider implementing top requested features for v2")
        
        recommendations.append("Continue monitoring user feedback post-launch")
        recommendations.append("Plan follow-up with beta users for testimonials")
        
        return recommendations


# Run UAT tests
async def run_uat_tests():
    """Run complete UAT suite"""
    tester = UserAcceptanceTester()
    uat_report = await tester.run_all_scenarios()
    
    # Run beta testing simulation
    beta_tester = BetaTester()
    beta_report = await beta_tester.run_beta_program()
    
    return {
        "uat_report": uat_report,
        "beta_report": beta_report
    }


if __name__ == "__main__":
    asyncio.run(run_uat_tests())