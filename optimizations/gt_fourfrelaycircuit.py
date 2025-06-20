import os
import sys

# Setting the path for XLuminA modules:
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)

if parent_path not in sys.path:
    sys.path.append(parent_path)

import xlumina.vectorized_optics as vo
import xlumina.optical_elements as oe
from xlumina import cm, mm, nm, um
import jax

def four_f_relay_circuit(input_light, parameters):

    """
    Define an optical table with a 4f system composed by 2 lenses, noise sources in the SLMs and misalignment. 

    [Distances must be input in cm]

    Parameters:
        input_light (VectorizedLight)
        parameters (list): Optical parameters [distance_1, distance_2, distance_3, lens_1_focal, lens_2_focal].

    Returns the detected light. 
    """

    light_stage0, _ = input_light.VRS_propagation(z=parameters[0]*cm)
    #light_stage0.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')
    modulated_lens1, _ = oe.lens(light_stage0, radius=(5/2*mm, 5/2*mm), focal=(parameters[3], parameters[3]))
    #modulated_lens1.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')
    light_stage1, _ = modulated_lens1.VRS_propagation(z=parameters[1]*cm)
    #light_stage1.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')
    modulated_lens2, _ = oe.lens(light_stage1, radius=(5/2*mm, 5/2*mm), focal=(parameters[4], parameters[4]))
    #modulated_lens2.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')
    detected_light, _ = modulated_lens2.VRS_propagation(z=parameters[2]*cm)
    #detected_light.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')

    return detected_light

def four_f_relay_circuit_scalar(input_light, parameters):

    """
    Define an optical table with a 4f system composed by 2 lenses, noise sources in the SLMs and misalignment. 

    [Distances must be input in cm]

    Parameters:
        input_light (ScalarLight)
        parameters (list): Optical parameters [distance_1, distance_2, distance_3, lens_1_focal, lens_2_focal].

    Returns the detected light. 
    """

    light_stage0, _ = input_light.RS_propagation(z=parameters[0]*cm)
    #light_stage0.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')
    modulated_lens1, _ = oe.lens(light_stage0, radius=(5/2*mm, 5/2*mm), focal=(parameters[3], parameters[3]))
    #modulated_lens1.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')
    light_stage1, _ = modulated_lens1.RS_propagation(z=parameters[1]*cm)
    #light_stage1.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')
    modulated_lens2, _ = oe.lens(light_stage1, radius=(5/2*mm, 5/2*mm), focal=(parameters[4], parameters[4]))
    #modulated_lens2.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')
    detected_light, _ = modulated_lens2.RS_propagation(z=parameters[2]*cm)
    #detected_light.draw(xlim=(-1000, 1000), ylim=(-1000, 1000), kind='Intensity')

    return detected_light


if __name__ == "__main__":

    import xlumina.toolbox as tb
    import jax.numpy as jnp
    
    sensor_lateral_size = 1024  # Resolution
    wavelength = 632.8*nm
    x_total = 1200*um # Space dimension
    x, y = tb.space(x_total, sensor_lateral_size)

    jones_vector = jnp.array([1, 1])
    w0 = (1200*um , 1200*um)
    gb0 = vo.PolarizedLightSource(x, y, wavelength)
    gb0.gaussian_beam(w0=w0, jones_vector=jones_vector)

    filename = 'optimizations/MPI_logo.png'
    MPI_mask = tb.image_to_binary_mask(filename, x, y, normalize=True, invert=True, threshold=0.5)

    key = jax.random.PRNGKey(42)

    focal_1 = 10*mm
    focal_2 = 10*mm
    d1 = 1 # cm
    d2 = 2 # cm
    d3 = 1 # cm

    #masking:
    gb0.Ex = gb0.Ex#* MPI_mask
    gb0.Ey = gb0.Ey#* MPI_mask
    gb0.Ez = gb0.Ez#* MPI_mask

    target_light = four_f_relay_circuit(gb0, [d1, d2, d3, focal_1, focal_2])

    gb0.draw(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), kind='Intensity')
    target_light.draw(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), kind='Intensity')

    