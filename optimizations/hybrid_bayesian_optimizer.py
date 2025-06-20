import os
import sys
import functools

# Setting the path for XLuminA modules:
current_path = os.path.abspath(os.path.join('.'))
if current_path not in sys.path:
    sys.path.append(current_path)

# Also add the parent directory (where xlumina is located)    
parent_path = os.path.abspath(os.path.join('..'))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import xlumina.vectorized_optics as vo
import xlumina.optical_elements as oe
from xlumina import cm, mm, nm, um
import jax
import xlumina.toolbox as tb
import jax.numpy as jnp
import optax
import numpy as np
import time
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class HybridBayesianOptimizer:
    """
    Hybrid optimizer that combines Bayesian optimization for global search 
    with gradient descent for local refinement.
    """
    
    def __init__(self, 
                 loss_function: Callable,
                 param_bounds: List[Tuple[float, float]],
                 circuit_function: Optional[Callable] = None,
                 verbose: bool = True):
        """
        Initialize the hybrid optimizer.
        
        Args:
            loss_function: The loss function to minimize
            param_bounds: List of (min, max) bounds for each parameter
            circuit_function: Optional circuit function for evaluation
            verbose: Whether to print progress
        """
        self.loss_function = loss_function
        self.param_bounds = param_bounds
        self.circuit_function = circuit_function
        self.verbose = verbose
        self.n_params = len(param_bounds)
        
        # Bayesian optimization history
        self.bo_X = []  # Parameter values tried
        self.bo_y = []  # Corresponding loss values
        
        # Best parameters found
        self.best_params = None
        self.best_loss = float('inf')
        self.optimization_history = {
            'bo_losses': [],
            'gd_losses': [],
            'bo_params': [],
            'gd_params': []
        }
    
    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1] range for Bayesian optimization."""
        normalized = np.zeros_like(params)
        for i, (low, high) in enumerate(self.param_bounds):
            normalized[i] = (params[i] - low) / (high - low)
        return normalized
    
    def _denormalize_params(self, normalized_params: np.ndarray) -> np.ndarray:
        """Convert normalized parameters back to original range."""
        params = np.zeros_like(normalized_params)
        for i, (low, high) in enumerate(self.param_bounds):
            params[i] = low + normalized_params[i] * (high - low)
        return params
    
    def _evaluate_loss(self, params: np.ndarray) -> float:
        """Evaluate the loss function at given parameters."""
        try:
            # Convert to JAX array if needed
            if isinstance(params, np.ndarray):
                params_jax = jnp.array(params)
            else:
                params_jax = params
                
            loss = float(self.loss_function(params_jax))
            
            # Update best parameters if this is better
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = params.copy()
                
            return loss
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating loss: {e}")
            return float('inf')
    
    def _acquisition_function(self, X_norm: np.ndarray, gp: GaussianProcessRegressor) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        """
        X_norm = X_norm.reshape(-1, self.n_params)
        mu, sigma = gp.predict(X_norm, return_std=True)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-9)
        
        # Current best (minimum) value
        f_best = np.min(self.bo_y) if self.bo_y else 0
        
        # Expected improvement
        with np.errstate(divide='ignore'):
            z = (f_best - mu) / sigma
            ei = (f_best - mu) * self._normal_cdf(z) + sigma * self._normal_pdf(z)
            
        # Set EI to 0 where sigma is very small
        ei[sigma < 1e-9] = 0
        
        return -ei  # Minimize (scipy minimizes, so we negate)
    
    def _normal_cdf(self, x):
        """Standard normal CDF."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _normal_pdf(self, x):
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def bayesian_optimization_phase(self, n_initial: int = 10, n_iterations: int = 50) -> Union[np.ndarray, None]:
        """
        Phase 1: Bayesian optimization for global exploration.
        
        Args:
            n_initial: Number of random initial points
            n_iterations: Number of Bayesian optimization iterations
            
        Returns:
            Best parameters found during Bayesian optimization
        """
        if self.verbose:
            print("=== PHASE 1: BAYESIAN OPTIMIZATION ===")
            print(f"Exploring parameter space with {n_initial} initial points and {n_iterations} BO iterations")
        
        # Generate initial random points in normalized space
        np.random.seed(42)  # For reproducibility
        X_norm_init = np.random.uniform(0, 1, (n_initial, self.n_params))
        
        # Evaluate initial points
        for i, x_norm in enumerate(X_norm_init):
            x_real = self._denormalize_params(np.array(x_norm))
            loss = self._evaluate_loss(x_real)
            
            self.bo_X.append(x_norm)
            self.bo_y.append(loss)
            self.optimization_history['bo_losses'].append(loss)
            self.optimization_history['bo_params'].append(x_real.copy())
            
            if self.verbose:
                print(f"Initial point {i+1}/{n_initial}: Loss = {loss:.6f}")
        
        # Bayesian optimization loop
        for iteration in range(n_iterations):
            if self.verbose:
                print(f"\nBO Iteration {iteration+1}/{n_iterations}")
                print(f"Current best loss: {self.best_loss:.6f}")
            
            # Fit Gaussian Process
            X_norm_array = np.array(self.bo_X)
            y_array = np.array(self.bo_y)
            
            # Use Matern kernel for better optimization landscapes
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10
            )
            
            try:
                gp.fit(X_norm_array, y_array)
                
                # Find next point to evaluate using acquisition function
                best_ei = float('inf')
                best_x_norm = None
                
                # Multiple random restarts for acquisition optimization
                for _ in range(20):
                    x0_norm = np.random.uniform(0, 1, self.n_params)
                    bounds_norm = [(0, 1) for _ in range(self.n_params)]
                    
                    result = minimize(
                        self._acquisition_function,
                        x0_norm,
                        args=(gp,),
                        bounds=bounds_norm,
                        method='L-BFGS-B'
                    )
                    
                    if result.success and result.fun < best_ei:
                        best_ei = result.fun
                        best_x_norm = result.x
                
                if best_x_norm is not None:
                    # Evaluate the new point
                    x_real = self._denormalize_params(best_x_norm)
                    loss = self._evaluate_loss(x_real)
                    
                    self.bo_X.append(best_x_norm)
                    self.bo_y.append(loss)
                    self.optimization_history['bo_losses'].append(loss)
                    self.optimization_history['bo_params'].append(x_real.copy())
                    
                    if self.verbose:
                        print(f"New point loss: {loss:.6f}")
                        if loss < self.best_loss:
                            print("*** New best point found! ***")
                else:
                    if self.verbose:
                        print("Failed to find next point, using random exploration")
                    # Fall back to random exploration
                    x_norm = np.random.uniform(0, 1, self.n_params)
                    x_real = self._denormalize_params(x_norm)
                    loss = self._evaluate_loss(x_real)
                    
                    self.bo_X.append(x_norm)
                    self.bo_y.append(loss)
                    self.optimization_history['bo_losses'].append(loss)
                    self.optimization_history['bo_params'].append(x_real.copy())
                    
            except Exception as e:
                if self.verbose:
                    print(f"GP fitting failed: {e}, using random exploration")
                # Fall back to random exploration
                x_norm = np.random.uniform(0, 1, self.n_params)
                x_real = self._denormalize_params(x_norm)
                loss = self._evaluate_loss(x_real)
                
                self.bo_X.append(x_norm)
                self.bo_y.append(loss)
                self.optimization_history['bo_losses'].append(loss)
                self.optimization_history['bo_params'].append(x_real.copy())
        
        if self.verbose:
            print(f"\n=== BAYESIAN OPTIMIZATION COMPLETE ===")
            print(f"Best loss found: {self.best_loss:.6f}")
            print(f"Best parameters: {self.best_params}")
        
        return self.best_params if self.best_params is not None else np.array([])
    
    def gradient_descent_phase(self, 
                             initial_params: np.ndarray,
                             n_iterations: int = 1000,
                             learning_rate: float = 0.01,
                             optimizer_type: str = 'adamw') -> Tuple[np.ndarray, List[float]]:
        """
        Phase 2: Gradient descent for local refinement.
        
        Args:
            initial_params: Starting parameters from Bayesian optimization
            n_iterations: Number of gradient descent iterations
            learning_rate: Learning rate for optimization
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
            
        Returns:
            Tuple of (best_parameters, loss_history)
        """
        if self.verbose:
            print("\n=== PHASE 2: GRADIENT DESCENT REFINEMENT ===")
            print(f"Starting from loss: {self.best_loss:.6f}")
            print(f"Using {optimizer_type} optimizer with lr={learning_rate}")
        
        # Convert to JAX arrays
        params = jnp.array(initial_params)
        
        # Choose optimizer
        if optimizer_type == 'adamw':
            optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=0.0001)
        elif optimizer_type == 'adam':
            optimizer = optax.adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optax.sgd(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Initialize optimizer state
        opt_state = optimizer.init(params)
        
        @jax.jit
        def update_step(params, opt_state):
            loss_value, grads = jax.value_and_grad(self.loss_function)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_value
        
        # Gradient descent loop
        loss_history = []
        best_gd_loss = self.best_loss
        best_gd_params = initial_params.copy()
        no_improvement_count = 0
        patience = 200  # Early stopping patience
        
        for step in range(n_iterations):
            params, opt_state, loss_value = update_step(params, opt_state)
            loss_float = float(loss_value)
            loss_history.append(loss_float)
            
            # Track best parameters
            if loss_float < best_gd_loss:
                best_gd_loss = loss_float
                best_gd_params = np.array(params)
                no_improvement_count = 0
                
                # Update global best
                if loss_float < self.best_loss:
                    self.best_loss = loss_float
                    self.best_params = best_gd_params.copy()
            else:
                no_improvement_count += 1
            
            # Store in history
            self.optimization_history['gd_losses'].append(loss_float)
            self.optimization_history['gd_params'].append(np.array(params))
            
            # Progress reporting
            if self.verbose and (step % 100 == 0 or step < 10):
                print(f"GD Step {step}: Loss = {loss_float:.6f}, Best = {best_gd_loss:.6f}")
            
            # Early stopping
            if no_improvement_count >= patience:
                if self.verbose:
                    print(f"Early stopping at step {step} (no improvement for {patience} steps)")
                break
        
        if self.verbose:
            print(f"\n=== GRADIENT DESCENT COMPLETE ===")
            print(f"Final loss: {best_gd_loss:.6f}")
            print(f"Improvement from BO: {(self.optimization_history['bo_losses'][-1] - best_gd_loss):.6f}")
        
        return best_gd_params, loss_history
    
    def optimize(self, 
                 bo_initial: int = 15,
                 bo_iterations: int = 40,
                 gd_iterations: int = 1000,
                 gd_learning_rate: float = 0.01,
                 optimizer_type: str = 'adamw') -> Dict[str, Any]:
        """
        Run the complete hybrid optimization.
        
        Args:
            bo_initial: Number of initial random points for BO
            bo_iterations: Number of Bayesian optimization iterations
            gd_iterations: Number of gradient descent iterations
            gd_learning_rate: Learning rate for gradient descent
            optimizer_type: Type of gradient optimizer
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Phase 1: Bayesian Optimization
        bo_best_params = self.bayesian_optimization_phase(bo_initial, bo_iterations)
        bo_time = time.time() - start_time
        
        # Phase 2: Gradient Descent
        gd_start_time = time.time()
        gd_best_params, gd_loss_history = self.gradient_descent_phase(
            bo_best_params, gd_iterations, gd_learning_rate, optimizer_type
        )
        gd_time = time.time() - gd_start_time
        total_time = time.time() - start_time
        
        # Prepare results
        results = {
            'best_params': self.best_params,
            'best_loss': self.best_loss,
            'bo_best_loss': min(self.optimization_history['bo_losses']),
            'gd_best_loss': min(gd_loss_history) if gd_loss_history else self.best_loss,
            'bo_time': bo_time,
            'gd_time': gd_time,
            'total_time': total_time,
            'bo_history': self.optimization_history['bo_losses'],
            'gd_history': gd_loss_history,
            'n_bo_evaluations': len(self.optimization_history['bo_losses']),
            'n_gd_evaluations': len(gd_loss_history)
        }
        
        if self.verbose:
            print(f"\n=== OPTIMIZATION COMPLETE ===")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"BO time: {bo_time:.2f}s ({bo_time/total_time*100:.1f}%)")
            print(f"GD time: {gd_time:.2f}s ({gd_time/total_time*100:.1f}%)")
            print(f"Final best loss: {self.best_loss:.6f}")
            print(f"Total function evaluations: {results['n_bo_evaluations'] + results['n_gd_evaluations']}")
        
        return results
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot the optimization history."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Bayesian Optimization progress
        ax1.plot(self.optimization_history['bo_losses'], 'b.-', linewidth=2, markersize=6)
        ax1.set_xlabel('BO Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Bayesian Optimization Progress')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Gradient Descent progress
        if self.optimization_history['gd_losses']:
            ax2.plot(self.optimization_history['gd_losses'], 'r.-', linewidth=1, markersize=2)
            ax2.set_xlabel('GD Iteration')
            ax2.set_ylabel('Loss')
            ax2.set_title('Gradient Descent Progress')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # Plot 3: Combined progress
        n_bo = len(self.optimization_history['bo_losses'])
        n_gd = len(self.optimization_history['gd_losses'])
        
        # BO phase
        ax3.plot(range(n_bo), self.optimization_history['bo_losses'], 'b.-', 
                linewidth=2, markersize=4, label='Bayesian Opt')
        
        # GD phase
        if n_gd > 0:
            gd_x = range(n_bo, n_bo + n_gd)
            ax3.plot(gd_x, self.optimization_history['gd_losses'], 'r-', 
                    linewidth=1, alpha=0.8, label='Gradient Descent')
        
        # Mark transition
        if n_gd > 0:
            ax3.axvline(x=n_bo, color='green', linestyle='--', alpha=0.7, label='BO→GD Transition')
        
        ax3.set_xlabel('Total Iterations')
        ax3.set_ylabel('Loss')
        ax3.set_title('Complete Optimization Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

# Example usage functions
def create_simple_loss_function(target_intensity):
    """Create a simple loss function for testing."""
    @jax.jit
    def loss_fn(params):
        # Simple quadratic loss for testing
        return jnp.sum((params - target_intensity)**2)
    
    return loss_fn

def run_hybrid_optimization_example():
    """Example of how to use the hybrid optimizer."""
    print("=== HYBRID BAYESIAN-GRADIENT OPTIMIZER EXAMPLE ===")
    
    # Create a simple test case
    target_params = jnp.array([2.0, 1.5, 3.0])  # Ground truth
    
    @jax.jit
    def test_loss(params):
        # Rosenbrock-like function with multiple local minima
        x, y, z = params[0], params[1], params[2]
        return 100*(y - x**2)**2 + (1 - x)**2 + 100*(z - y**2)**2 + (1 - y)**2
    
    # Define parameter bounds
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    # Create and run optimizer
    optimizer = HybridBayesianOptimizer(
        loss_function=test_loss,
        param_bounds=bounds,
        verbose=True
    )
    
    results = optimizer.optimize(
        bo_initial=10,
        bo_iterations=20,
        gd_iterations=500,
        gd_learning_rate=0.01
    )
    
    # Plot results
    optimizer.plot_optimization_history()
    
    print(f"Ground truth: {target_params}")
    print(f"Found solution: {results['best_params']}")
    print(f"Error: {jnp.linalg.norm(results['best_params'] - target_params):.6f}")
    
    return results

if __name__ == "__main__":
    # Run example
    results = run_hybrid_optimization_example() 