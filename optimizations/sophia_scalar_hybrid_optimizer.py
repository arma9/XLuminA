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

import xlumina.wave_optics as so  # Use scalar optics instead of vectorized
import xlumina.optical_elements as oe
from xlumina import cm, mm, nm, um
import jax
import xlumina.toolbox as tb
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt

from gt_fourfrelaycircuit import four_f_relay_circuit_scalar
from hybrid_bayesian_optimizer import HybridBayesianOptimizer

def setup_optical_system():
    """Setup the optical system exactly as in the sophia notebook using scalar optics."""
    print("Setting up optical system with scalar optics...")

    sensor_lateral_size = 1024  # Resolution
    wavelength = 632.8*nm
    x_total = 1200*um # Space dimension
    x, y = tb.space(x_total, sensor_lateral_size)

    w0 = (1200*um , 1200*um)
    gb_scalar = so.LightSource(x, y, wavelength)
    gb_scalar.gaussian_beam(w0=w0, E0=1)

    #filename = 'MPI_logo.png'
    #MPI_mask = tb.image_to_binary_mask(filename, x, y, normalize=True, invert=True, threshold=0.5)
    #gb_scalar.field = gb_scalar.field * MPI_mask
    
    input_light = gb_scalar

    # Target setup (ground truth parameters)
    focal_1 = 10*mm
    focal_2 = 10*mm
    d1 = 1  # cm
    d2 = 2  # cm  
    d3 = 1  # cm (this is the parameter we're optimizing for)

    target_light = four_f_relay_circuit_scalar(input_light, [d1, d2, d3, focal_1, focal_2])
    
    def get_intensity(light):
        if isinstance(light, so.ScalarLight):
            intensity = jnp.abs(light.field)**2
        else:
            intensity = jnp.abs(light.field)**2
        return jnp.array(intensity)

    target_intensity = get_intensity(target_light)
    
    print(f"Target parameters: d1={d1}, d2={d2}, d3={d3} (optimizing d3)")
    print(f"Target intensity shape: {target_intensity.shape}")
    
    return input_light, target_intensity, x, y

def create_circuit_function(input_light):
    """Create the circuit function that we want to optimize using scalar optics."""
    
    @functools.partial(jax.jit, static_argnums=(1,))  # Mark light (argument 1) as static
    def scalar_circuit(parameters, light):
        # Extract parameters - we're optimizing d_3 (distance 3)
        if jnp.ndim(parameters) == 0:
            # Single parameter case
            d_3 = parameters
        else:
            # Multiple parameters case (for future extension)
            d_3 = parameters[0] if len(parameters) > 0 else parameters
            
        # Fixed parameters from the notebook (must match target setup!)
        d_1 = 1  # cm
        d_2 = 2  # cm  
        f_1 = 10  # mm  
        f_2 = 10  # mm  

        # Execute the circuit using scalar propagation
        light_stage0, _ = light.RS_propagation(z=d_1*cm)
        modulated_lens1, _ = oe.lens(light_stage0, radius=(5/2*mm, 5/2*mm), focal=(f_1, f_1))
        light_stage1, _ = modulated_lens1.RS_propagation(z=d_2*cm)
        modulated_lens2, _ = oe.lens(light_stage1, radius=(5/2*mm, 5/2*mm), focal=(f_2, f_2))
        detected_light, _ = modulated_lens2.RS_propagation(z=d_3*cm)
        
        # Calculate intensity for scalar field
        intensity = jnp.array(jnp.abs(detected_light.field)**2)
        return intensity

    circuit_partial = functools.partial(scalar_circuit, light=input_light)
    return circuit_partial

def create_loss_function(circuit_function, target_intensity):
    """Create the loss function for optimization."""
    
    def loss_sophia_scalar(parameters):
        input_intensity = circuit_function(parameters)
        mse = jnp.sum((input_intensity - target_intensity)**2) 
        return mse

    return jax.jit(loss_sophia_scalar)

def run_sophia_scalar_hybrid_optimization():
    """Run the complete hybrid optimization on the Sophia circuit using scalar optics."""
    print("=== SOPHIA CIRCUIT SCALAR HYBRID OPTIMIZATION ===")
    
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
        gd_iterations=50,     # Sufficient GD iterations
        gd_learning_rate=0.001, # Learning rate from notebook
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
    optimizer.plot_optimization_history(save_path='sophia_scalar_optimization_history.png')
    
    # Visualize the results
    visualize_results(input_light, target_intensity, results['best_params'][0], x, y)
    
    return results

def visualize_results(input_light, target_intensity, optimized_d3, x, y):
    """Visualize the optimization results for scalar optics."""
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
    axes[0].set_title('Target Intensity (Scalar)')
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
    plt.savefig('sophia_scalar_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print similarity metrics
    mse = jnp.mean((target_intensity - optimized_intensity)**2)
    max_target = jnp.max(target_intensity)
    max_optimized = jnp.max(optimized_intensity)
    
    print(f"MSE between target and optimized: {mse:.6f}")
    print(f"Max target intensity: {max_target:.6f}")
    print(f"Max optimized intensity: {max_optimized:.6f}")
    print(f"Relative error: {jnp.sqrt(mse) / max_target * 100:.2f}%")

def test_parameter_sensitivity():
    """Test the sensitivity of the loss function to parameter changes."""
    print("\n=== PARAMETER SENSITIVITY TEST ===")
    
    # Setup system
    input_light, target_intensity, x, y = setup_optical_system()
    circuit_function = create_circuit_function(input_light)
    loss_function = create_loss_function(circuit_function, target_intensity)
    
    # Test range of d_3 values around the ground truth (1.0 cm)
    d3_values = jnp.linspace(0.5, 2.0, 50)
    losses = []
    
    print("Computing loss landscape...")
    for d3 in d3_values:
        loss = loss_function(jnp.array([d3]))
        losses.append(float(loss))
    
    losses = jnp.array(losses)
    
    # Plot the loss landscape
    plt.figure(figsize=(10, 6))
    plt.plot(d3_values, losses, 'b-', linewidth=2)
    plt.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Ground Truth (d3=1.0)')
    plt.xlabel('d3 (cm)')
    plt.ylabel('Loss')
    plt.title('Loss Landscape for d3 Parameter (Scalar Optics)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('sophia_scalar_loss_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find minimum
    min_idx = jnp.argmin(losses)
    optimal_d3 = d3_values[min_idx]
    min_loss = losses[min_idx]
    
    print(f"Ground truth d3: 1.0 cm")
    print(f"Loss landscape minimum at d3: {optimal_d3:.6f} cm")
    print(f"Minimum loss: {min_loss:.6f}")
    print(f"Error from ground truth: {abs(optimal_d3 - 1.0):.6f} cm")

def optimize_scalar_simple(params: optax.Params, optimizer: optax.GradientTransformation, num_iterations, loss_function):
    """Simple scalar optimization function similar to the notebook."""
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss_function, allow_int=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    iteration_steps = []
    loss_list = []

    n_best = 500
    best_loss = 1e2
    best_params = None
    best_step = 0
    
    print('Starting Simple Scalar Optimization', flush=True)
    
    for step in range(num_iterations):        
        params, opt_state, loss_value = update(params, opt_state)
        
        if step % 50 == 0:
            print(f"Step {step}")
            print(f"Loss {loss_value}")
            print(f"Parameters {params}")
        
        iteration_steps.append(step)
        loss_list.append(loss_value)
        
        # Update the `best_loss` value:
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = params
            best_step = step
            if step % 50 == 0:
                print('Best loss value is updated')
            
        if step % 100 == 0:
            # Stopping criteria: if best_loss has not changed every 500 steps, stop.
            if step - best_step > n_best:
                print(f'Stopping criterion: no improvement in loss value for {n_best} steps')
                break
    
    print(f'Best loss: {best_loss} at step {best_step}')
    print(f'Best parameters: {best_params}')  
    return best_params, best_loss, iteration_steps, loss_list

def run_simple_scalar_optimization():
    """Run simple scalar optimization like in the notebook."""
    print("=== SIMPLE SCALAR OPTIMIZATION ===")
    
    # Setup system
    input_light, target_intensity, x, y = setup_optical_system()
    circuit_function = create_circuit_function(input_light)
    loss_function = create_loss_function(circuit_function, target_intensity)
    
    # Set up optimization
    params = jnp.array(10.0)  # Starting point from notebook
    optimizer = optax.adamw(learning_rate=0.5)  # Same as notebook
    num_iterations = 1000
    
    # Run optimization
    best_params, best_loss, iteration_steps, loss_list = optimize_scalar_simple(
        params, optimizer, num_iterations, loss_function
    )
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_steps, loss_list, 'b-', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Simple Scalar Optimization Convergence')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('sophia_scalar_simple_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== SIMPLE OPTIMIZATION RESULTS ===")
    print(f"Ground truth d_3: 1.0 cm")
    print(f"Found d_3: {best_params:.6f} cm")
    print(f"Error: {abs(best_params - 1.0):.6f} cm")
    print(f"Final loss: {best_loss:.6f}")
    
    return best_params, best_loss

if __name__ == "__main__":
    print("Sophia Scalar Hybrid Optimizer")
    print("=" * 50)
    

    print("Running complete hybrid optimization...")
    hybrid_result = run_sophia_scalar_hybrid_optimization()
    
    print("\n=== FINAL RESULTS ===")
    print(f"Hybrid optimization result: {hybrid_result['best_params'][0]:.6f} cm")
    print(f"Ground truth: 1.0 cm")
    print(f"Error: {abs(hybrid_result['best_params'][0] - 1.0):.6f} cm")
    print(f"Final loss: {hybrid_result['best_loss']:.6f}")
    print(f"Total optimization time: {hybrid_result['total_time']:.2f} seconds") 