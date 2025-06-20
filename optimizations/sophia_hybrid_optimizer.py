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
import matplotlib.pyplot as plt

from gt_fourfrelaycircuit import four_f_relay_circuit
from hybrid_bayesian_optimizer import HybridBayesianOptimizer

def setup_optical_system():
    """Setup the optical system exactly as in the sophia notebook."""
    print("Setting up optical system...")

    sensor_lateral_size = 1024  # Resolution
    wavelength = 632.8*nm
    x_total = 1200*um # Space dimension
    x, y = tb.space(x_total, sensor_lateral_size)

    jones_vector = jnp.array([1, 1])
    w0 = (1200*um , 1200*um)
    gb0 = vo.PolarizedLightSource(x, y, wavelength)
    gb0.gaussian_beam(w0=w0, jones_vector=jones_vector)

    #filename = 'MPI_logo.png'

    # MPI_mask = tb.image_to_binary_mask(filename, x, y, normalize=True, invert=True, threshold=0.5)
    # gb0.Ex = gb0.Ex * MPI_mask
    # gb0.Ey = gb0.Ey * MPI_mask  
    # gb0.Ez = gb0.Ez * MPI_mask
    
    input_light = gb0

    # Target setup (ground truth parameters)
    focal_1 = 10*mm
    focal_2 = 10*mm
    d1 = 1  # cm
    d2 = 2  # cm  
    d3 = 1  # cm (this is the parameter we're optimizing for)

    target_light = four_f_relay_circuit(input_light, [d1, d2, d3, focal_1, focal_2])
    
    def get_intensity(light):
        intensity = jnp.abs(light.Ex)**2 + jnp.abs(light.Ey)**2
        return jnp.array(intensity)

    target_intensity = get_intensity(target_light)
    
    print(f"Target parameters: d1={d1}, d2={d2}, d3={d3} (optimizing d3)")
    print(f"Target intensity shape: {target_intensity.shape}")
    
    return input_light, target_intensity, x, y

def create_circuit_function(input_light):
    """Create the circuit function that we want to optimize."""
    
    @functools.partial(jax.jit, static_argnums=(1,))  # Mark light (argument 1) as static
    def circuit(parameters, light):
        # Extract parameters - we're optimizing d_3 (distance 3)
        if jnp.ndim(parameters) == 0:
            # Single parameter case
            d_3 = parameters
        else:
            # Multiple parameters case (for future extension)
            d_3 = parameters[0] if len(parameters) > 0 else parameters
            
        # Fixed parameters from the notebook (must match target setup!)
        d_1 = 1  # cm
        d_2 = 2  # cm  # FIXED: was 4.5, now matches target setup
        f_1 = 10  # mm  # matches target focal_1 = 10*mm
        f_2 = 10  # mm  # FIXED: was 20, now matches target focal_2 = 10*mm

        # Execute the circuit
        light_stage0, _ = light.VRS_propagation(z=d_1*cm)
        modulated_lens1, _ = oe.lens(light_stage0, radius=(5/2*mm, 5/2*mm), focal=(f_1, f_1))
        light_stage1, _ = modulated_lens1.VRS_propagation(z=d_2*cm)
        modulated_lens2, _ = oe.lens(light_stage1, radius=(5/2*mm, 5/2*mm), focal=(f_2, f_2))
        detected_light, _ = modulated_lens2.VRS_propagation(z=d_3*cm)
        
        # Calculate intensity
        intensity = jnp.array(jnp.abs(detected_light.Ex)**2 + jnp.abs(detected_light.Ey)**2)
        return intensity

    circuit_partial = functools.partial(circuit, light=input_light)
    return circuit_partial

def create_loss_function(circuit_function, target_intensity):
    """Create the loss function for optimization."""
    
    def loss_sophia(parameters):
        input_intensity = circuit_function(parameters)
        mse = jnp.sum((input_intensity - target_intensity)**2) 
        return mse

    return jax.jit(loss_sophia)

def run_sophia_hybrid_optimization():
    """Run the complete hybrid optimization on the Sophia circuit."""
    print("=== SOPHIA CIRCUIT HYBRID OPTIMIZATION ===")
    
    # Setup optical system
    input_light, target_intensity, x, y = setup_optical_system()
    
    # Create circuit and loss functions
    circuit_function = create_circuit_function(input_light)
    loss_function = create_loss_function(circuit_function, target_intensity)
    
    # Test the loss function with initial parameter
    initial_param = 10.0  # From notebook
    initial_loss = loss_function(jnp.array([initial_param]))
    print(f"Initial parameter: {initial_param}")
    print(f"Initial loss: {initial_loss}")
    
    # Define parameter bounds for d_3 (distance in cm)
    # Reasonable range for optical distances
    param_bounds = [(0.1, 20.0)]  # d_3 can range from 0.1 cm to 20 cm
    
    print(f"Parameter bounds: {param_bounds}")
    print(f"Ground truth d_3: 1.0 cm (from target setup)")
    
    # Create and run hybrid optimizer
    optimizer = HybridBayesianOptimizer(
        loss_function=loss_function,
        param_bounds=param_bounds,
        circuit_function=circuit_function,
        verbose=True
    )
    
    # Run optimization with appropriate settings for this problem
    results = optimizer.optimize(
        bo_initial=12,         # More initial points for better coverage
        bo_iterations=30,      # Reasonable number of BO iterations  
        gd_iterations=800,     # Sufficient GD iterations
        gd_learning_rate=0.05, # Learning rate from notebook
        optimizer_type='adamw' # Same as notebook
    )
    
    # Print results
    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"Ground truth d_3: 1.0 cm")
    print(f"Found d_3: {results['best_params'][0]:.6f} cm")
    print(f"Error: {abs(results['best_params'][0] - 1.0):.6f} cm")
    print(f"Final loss: {results['best_loss']:.6f}")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Improvement: {initial_loss - results['best_loss']:.6f}")
    print(f"Optimization time: {results['total_time']:.2f} seconds")
    
    # Plot optimization history
    optimizer.plot_optimization_history(save_path='sophia_optimization_history.png')
    
    # Visualize the results
    visualize_results(input_light, target_intensity, results['best_params'][0], x, y)
    
    return results

def visualize_results(input_light, target_intensity, optimized_d3, x, y):
    """Visualize the optimization results."""
    print("\n=== VISUALIZATION ===")
    
    # Create circuit with optimized parameters
    circuit_function = create_circuit_function(input_light)
    optimized_intensity = circuit_function(jnp.array([optimized_d3]))
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    xlim = 500
    ylim = 500
    
    # Plot target intensity
    extent = [-xlim, xlim, -ylim, ylim]
    im1 = axes[0].imshow(target_intensity, extent=extent, cmap='hot')
    axes[0].set_title('Target Intensity')
    axes[0].set_xlabel('x (μm)')
    axes[0].set_ylabel('y (μm)')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot optimized intensity
    im2 = axes[1].imshow(optimized_intensity, extent=extent, cmap='hot')
    axes[1].set_title(f'Optimized Intensity (d3={optimized_d3:.3f})')
    axes[1].set_xlabel('x (μm)')
    axes[1].set_ylabel('y (μm)')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot difference
    difference = jnp.abs(target_intensity - optimized_intensity)
    im3 = axes[2].imshow(difference, extent=extent, cmap='viridis')
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('x (μm)')
    axes[2].set_ylabel('y (μm)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('sophia_results_comparison.png', dpi=300, bbox_inches='tight')
    print("Results saved to 'sophia_results_comparison.png'")
    #plt.show() 

def test_parameter_sensitivity():
    """Test how sensitive the loss is to parameter changes."""
    print("\n=== PARAMETER SENSITIVITY ANALYSIS ===")
    
    # Setup system
    input_light, target_intensity, x, y = setup_optical_system()
    circuit_function = create_circuit_function(input_light)
    loss_function = create_loss_function(circuit_function, target_intensity)
    
    # Test range of parameters around the expected minimum at d3=1.0
    d3_values = jnp.linspace(0.2, 2.0, 25)
    losses = []
    
    print("Testing parameter sensitivity...")
    for d3 in d3_values:
        loss = loss_function(jnp.array([float(d3)]))
        losses.append(float(loss))
        print(f"d3 = {d3:.2f} cm, loss = {loss:.2e}")
    
    # Plot sensitivity
    plt.figure(figsize=(10, 6))
    plt.plot(d3_values, losses, 'b.-', linewidth=2, markersize=6)
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Ground Truth (d3=1.0)')
    plt.xlabel('d3 (cm)')
    plt.ylabel('Loss')
    plt.title('Parameter Sensitivity Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print("Sensitivity plot saved to 'parameter_sensitivity.png'")
    #plt.show()  # Commented out to avoid blocking
    
    return d3_values, losses

if __name__ == "__main__":
    print("Starting Sophia Circuit Hybrid Optimization...")
    
    # Run parameter sensitivity analysis first
    test_parameter_sensitivity()
    
    # Run the main optimization
    results = run_sophia_hybrid_optimization()
    
    print(f"\n=== SUMMARY ===")
    print(f"The hybrid optimization successfully found d3 = {results['best_params'][0]:.6f} cm")
    print(f"Ground truth was d3 = 1.0 cm")
    print(f"Error: {abs(results['best_params'][0] - 1.0):.6f} cm ({abs(results['best_params'][0] - 1.0)/1.0*100:.2f}%)")
    print(f"Total time: {results['total_time']:.1f} seconds")
    print(f"BO found: {results['bo_best_loss']:.2e}, GD refined to: {results['gd_best_loss']:.2e}") 