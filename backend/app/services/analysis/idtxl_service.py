import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Patch numpy.math for IDTxl compatibility with newer numpy versions
if not hasattr(np, 'math'):
    np.math = math

from idtxl.multivariate_te import MultivariateTE
from idtxl.multivariate_mi import MultivariateMI
from idtxl.data import Data
from idtxl.bivariate_te import BivariateTE
from idtxl.bivariate_mi import BivariateMI

from app.models.analysis import IDTxlConfig, IDTxlResult
from app.models.data import TimeSeriesData


class IDTxlService:
    """Service wrapper for IDTxl information-theoretic analysis"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def analyze(
        self, 
        time_series_data: Dict[str, TimeSeriesData], 
        config: IDTxlConfig
    ) -> IDTxlResult:
        """Perform IDTxl analysis on financial time series data"""
        
        # Prepare data for IDTxl
        data_array, n_processes = self._prepare_data(time_series_data, config.variables)
        
        # Create IDTxl Data object
        # IDTxl expects data in format: data[process, samples]
        idtxl_data = Data(data_array, dim_order='ps')
        
        # Run analysis based on configuration
        loop = asyncio.get_event_loop()
        
        results = {}
        processing_time = 0
        
        if config.analysis_type in ["transfer_entropy", "both"]:
            te_result, te_time = await loop.run_in_executor(
                self.executor,
                self._run_transfer_entropy,
                idtxl_data,
                config
            )
            results["transfer_entropy"] = te_result
            processing_time += te_time
        
        if config.analysis_type in ["mutual_information", "both"]:
            mi_result, mi_time = await loop.run_in_executor(
                self.executor,
                self._run_mutual_information,
                idtxl_data,
                config
            )
            results["mutual_information"] = mi_result
            processing_time += mi_time
        
        # Extract significant connections
        significant_connections = self._extract_significant_connections(results)
        
        return IDTxlResult(
            transfer_entropy=results.get("transfer_entropy"),
            mutual_information=results.get("mutual_information"),
            significant_connections=significant_connections,
            processing_time=processing_time
        )
    
    def _prepare_data(
        self, 
        time_series_data: Dict[str, TimeSeriesData], 
        variables: List[str]
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """Prepare data for IDTxl analysis"""
        
        # Extract close prices for each variable
        data_arrays = []
        for var in variables:
            if var not in time_series_data:
                raise ValueError(f"Variable {var} not found in time series data")
            
            # Extract close prices
            prices = [ohlcv.close for ohlcv in time_series_data[var].data]
            
            # Convert to returns (percentage change)
            returns = np.array(prices[1:]) / np.array(prices[:-1]) - 1
            
            # Standardize
            returns = (returns - np.mean(returns)) / np.std(returns)
            
            data_arrays.append(returns)
        
        # Stack arrays
        # IDTxl expects shape (n_processes, n_samples)
        data_array = np.stack(data_arrays)
        
        # Return array directly (IDTxl Data class will handle it)
        n_processes = len(variables)
        
        return data_array, n_processes
    
    def _run_transfer_entropy(
        self, 
        data: Data, 
        config: IDTxlConfig
    ) -> Tuple[Dict[str, Any], float]:
        """Run transfer entropy analysis"""
        import time
        start_time = time.time()
        
        # Set up analysis
        if data.n_processes > 2:
            # Use multivariate TE for more than 2 variables
            network_analysis = MultivariateTE()
        else:
            # Use bivariate TE for 2 variables
            network_analysis = BivariateTE()
        
        # Configure settings - use JIDT estimators which are available
        estimator_map = {
            'kraskov': 'JidtKraskovCMI',
            'gaussian': 'JidtGaussianCMI',
            'symbolic': 'symbolic'
        }
        
        settings = {
            'cmi_estimator': estimator_map.get(config.estimator.value, 'JidtGaussianCMI'),
            'max_lag_sources': config.max_lag,
            'min_lag_sources': 1,
            'n_perm_max_stat': config.permutations,
            'alpha_max_stat': config.significance_level,
            'permute_in_time': True  # Essential for financial time series (single replication)
        }
        
        # Add estimator-specific settings
        if config.estimator.value == 'kraskov':
            settings['kraskov_k'] = config.k_neighbors or 3
            settings['noise_level'] = config.noise_level or 1e-8
        elif config.estimator.value == 'symbolic':
            settings['alph_sizes'] = config.alphabet_size or 2
        
        # Run analysis for each target
        results = {
            "connections": [],
            "network_summary": {},
            "settings": settings
        }
        
        try:
            for target in range(data.n_processes):
                # Define sources (all except target)
                sources = [i for i in range(data.n_processes) if i != target]
                
                if isinstance(network_analysis, MultivariateTE):
                    result = network_analysis.analyse_network(
                        settings=settings,
                        data=data,
                        targets=[target],
                        sources=sources
                    )
                else:
                    # For bivariate, analyze each source separately
                    for source in sources:
                        result = network_analysis.analyse_single_target(
                            settings=settings,
                            data=data,
                            target=target,
                            sources=[source]
                        )
                
                # Extract significant connections
                if result and hasattr(result, 'targets_analysed'):
                    for target_idx in result.targets_analysed:
                        sources = result.get_target_sources(target_idx)
                        if sources is not None and sources.size > 0:
                            # Get TE result for this target
                            te_result = result.get_single_target(target_idx, fdr=False)
                            if hasattr(te_result, 'te') and te_result.te is not None:
                                for source_idx in sources:
                                    # Try to get delay/lag information
                                    try:
                                        # Get selected variables to find the lag
                                        selected_vars = te_result.selected_vars_sources
                                        lag = 1  # Default lag
                                        # Find the lag for this source
                                        for var in selected_vars:
                                            if var[0] == source_idx:
                                                lag = var[1]
                                                break
                                    except:
                                        lag = 1  # Default if we can't extract lag
                                    
                                    results["connections"].append({
                                        "source": config.variables[int(source_idx)],
                                        "target": config.variables[int(target_idx)],
                                        "te_value": float(te_result.te),
                                        "lag": int(lag)
                                    })
        
        except Exception as e:
            print(f"Transfer entropy analysis error: {str(e)}")
            results["error"] = str(e)
        
        processing_time = time.time() - start_time
        return results, processing_time
    
    def _run_mutual_information(
        self, 
        data: Data, 
        config: IDTxlConfig
    ) -> Tuple[Dict[str, Any], float]:
        """Run mutual information analysis"""
        import time
        start_time = time.time()
        
        # Set up analysis
        if data.n_processes > 2:
            network_analysis = MultivariateMI()
        else:
            network_analysis = BivariateMI()
        
        # Configure settings - use JIDT estimators
        estimator_map = {
            'kraskov': 'JidtKraskovCMI',
            'gaussian': 'JidtGaussianCMI',
            'symbolic': 'symbolic'
        }
        
        settings = {
            'cmi_estimator': estimator_map.get(config.estimator.value, 'JidtGaussianCMI'),
            'max_lag_sources': config.max_lag,
            'min_lag_sources': 0,
            'n_perm_max_stat': config.permutations,
            'alpha_max_stat': config.significance_level,
            'permute_in_time': True  # Essential for financial time series (single replication)
        }
        
        # Add estimator-specific settings
        if config.estimator.value == 'kraskov':
            settings['kraskov_k'] = config.k_neighbors or 3
            settings['noise_level'] = config.noise_level or 1e-8
        
        # Initialize results
        n_vars = len(config.variables)
        mi_matrix = np.zeros((n_vars, n_vars))
        
        results = {
            "matrix": [],
            "variables": config.variables,
            "significant_pairs": []
        }
        
        try:
            # Calculate MI for each pair
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if isinstance(network_analysis, MultivariateMI):
                        result = network_analysis.analyse_network(
                            settings=settings,
                            data=data,
                            targets=[i],
                            sources=[j]
                        )
                    else:
                        result = network_analysis.analyse_single_target(
                            settings=settings,
                            data=data,
                            target=i,
                            sources=[j]
                        )
                    
                    # Extract MI value
                    if result and hasattr(result, 'get_single_target'):
                        mi_result = result.get_single_target(i, fdr=False)
                        if hasattr(mi_result, 'mi') and mi_result.mi is not None:
                            # Handle numpy array or scalar
                            mi_val = mi_result.mi
                            if hasattr(mi_val, 'item'):
                                mi_value = float(mi_val.item())
                            else:
                                mi_value = float(mi_val)
                            
                            mi_matrix[i, j] = mi_value
                            mi_matrix[j, i] = mi_value  # Symmetric
                            
                            # Check if significant
                            if hasattr(mi_result, 'selected_vars_sources') and len(mi_result.selected_vars_sources) > 0:
                                results["significant_pairs"].append({
                                    "var1": config.variables[i],
                                    "var2": config.variables[j],
                                    "mi_value": mi_value
                                })
            
            # Set diagonal to 1 (self-information)
            np.fill_diagonal(mi_matrix, 1.0)
            
            # Convert matrix to list
            results["matrix"] = mi_matrix.tolist()
            
        except Exception as e:
            print(f"Mutual information analysis error: {str(e)}")
            results["error"] = str(e)
        
        processing_time = time.time() - start_time
        return results, processing_time
    
    def _extract_significant_connections(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all significant connections from results"""
        connections = []
        
        # Extract from transfer entropy
        if "transfer_entropy" in results and "connections" in results["transfer_entropy"]:
            for conn in results["transfer_entropy"]["connections"]:
                connections.append({
                    "type": "transfer_entropy",
                    "source": conn["source"],
                    "target": conn["target"],
                    "strength": conn["te_value"],
                    "lag": conn.get("lag", 1)
                })
        
        # Extract from mutual information
        if "mutual_information" in results and "significant_pairs" in results["mutual_information"]:
            for pair in results["mutual_information"]["significant_pairs"]:
                connections.append({
                    "type": "mutual_information",
                    "source": pair["var1"],
                    "target": pair["var2"],
                    "strength": pair["mi_value"],
                    "lag": 0  # MI is instantaneous
                })
        
        # Sort by strength
        connections.sort(key=lambda x: x["strength"], reverse=True)
        
        return connections