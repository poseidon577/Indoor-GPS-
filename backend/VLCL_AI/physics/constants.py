# constants.py
import math

# Physical Constants
SPEED_OF_LIGHT = 299792458.0  # m/s
ELECTRON_CHARGE = 1.602176634e-19  # C
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
PLANCK_CONSTANT = 6.62607015e-34  # J*s

# Optical / LED Constants
DEFAULT_WAVELENGTH = 450e-9  # m (Blue LED peak)
DEFAULT_REFRACTIVE_INDEX = 1.5  # Refractive index of optical concentrator

# Photodiode / APD Constants
DEFAULT_RESPONSIVITY = 0.54  # A/W
DEFAULT_RECEIVER_AREA = 1e-4  # m^2 (1 cm^2 = 1e-4 m^2; note: 1 mm^2 = 1e-6 m^2, which is different)
DEFAULT_DARK_CURRENT = 1e-9  # A (Dark current of photodiode)
DEFAULT_CAPACITANCE = 5e-12  # F (Photodiode junction capacitance)
DEFAULT_BANDWIDTH = 20e6  # Hz (20 MHz)
DEFAULT_TRANSIMPEDANCE_GAIN = 1e4  # V/A (TIA Ohm gain)

# Environmental / Channel Constants
DEFAULT_AMBIENT_TEMPERATURE = 298.15  # K (25 degrees C)
DEFAULT_BACKGROUND_CURRENT = 100e-6  # A (Induced background current from ambient light)
DEFAULT_ATMOSPHERIC_LOSS = 0.001  # dB/m
DEFAULT_LENS_GAIN = 1.5
