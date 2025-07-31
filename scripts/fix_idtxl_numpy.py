"""Fix IDTxl numpy compatibility issue"""
import numpy as np
import math

# Monkey patch numpy.math for IDTxl compatibility
if not hasattr(np, 'math'):
    np.math = math
    print("Patched numpy.math for IDTxl compatibility")

# Now we can use IDTxl
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

print("\nTesting IDTxl with numpy patch...")
print("=" * 50)

# Create simple test data
data = Data()
data.generate_mute_data(n_samples=100, n_replications=3)
print(f"Generated data: {data.n_processes} processes, {data.n_samples} samples")

# Run analysis
network_analysis = MultivariateTE()
settings = {
    'cmi_estimator': 'JidtGaussianCMI',
    'max_lag_sources': 3,
    'min_lag_sources': 1,
    'n_perm_max_stat': 50,
    'alpha_max_stat': 0.05
}

try:
    print("\nRunning network analysis...")
    results = network_analysis.analyse_network(settings=settings, data=data)
    print("Analysis completed successfully!")
    
    # Check results
    targets_analyzed = results.targets_analysed
    print(f"\nTargets analyzed: {targets_analyzed}")
    
    for target in targets_analyzed:
        selected_sources = results.get_target_sources(target)
        if selected_sources:
            print(f"Target {target} sources: {selected_sources}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()