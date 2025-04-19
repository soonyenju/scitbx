import numpy as np

# # Define the Arrhenius equation for respiration
# def arrhenius_temperature_dependence(T, E_a, R=8.314, T_0=293.15):
#     """
#     Calculate temperature-dependent respiration using Arrhenius equation.
#     T: temperature (in Kelvin)
#     E_a: activation energy (J/mol)
#     R: universal gas constant (J/mol·K)
#     T_0: reference temperature (in Kelvin)
#     """
#     return np.exp(E_a / R * (1 / T - 1 / T_0))

# # Heterotrophic respiration function (simple version)
# def heterotrophic_respiration(T, E_h, moisture_factor=1, R=8.314, T_0=293.15):
#     """
#     Calculate heterotrophic respiration using temperature dependence and moisture factor.
#     T: temperature in Celsius
#     E_h: activation energy for heterotrophic respiration (J/mol)
#     moisture_factor: scaling factor for moisture (default = 1, can be adjusted)
#     """
#     T_K = T + 273.15  # Convert to Kelvin
#     return arrhenius_temperature_dependence(T_K, E_h, R, T_0) * moisture_factor

# # Example function to calculate autotrophic respiration
# def calculate_autotrophic_respiration(T, E_a=60000):
#     """
#     Calculate autotrophic respiration from temperature.
#     T: temperature in Celsius
#     E_a: activation energy for autotrophic respiration (J/mol)
#     """
#     T_K = T + 273.15  # Convert to Kelvin
#     return arrhenius_temperature_dependence(T_K, E_a)

# def calculate_ecosystem_respiration_obsolete(T, E_a=60000, E_h=60000, moisture_factor = 1, R = 8.314, T_0 = 293.15):
#     """
#     Calculate the total ecosystem respiration using autotrophic and heterotrophic respiration.
#     T: temperature in Celsius
#     E_a: activation energy for autotrophic respiration (J/mol)
#     E_h: activation energy for heterotrophic respiration (J/mol)
#     moisture_factor: scaling factor for moisture (for heterotrophic, default = 1, can be adjusted) 
#     R: universal gas constant (J/mol·K)
#     T_0: reference temperature (in Kelvin)

#     Ea = 0.65 eV = ca.62.4 kJ/mol
#     Biome-scale temperature sensitivity of ecosystem respiration revealed by atmospheric CO2 observations
#     """
#     Ra = calculate_autotrophic_respiration(T, E_a)
#     Rh = heterotrophic_respiration(T, E_h, moisture_factor, R, T_0)
#     Reco = Ra + Rh
    
#     return Reco


def calculate_ecosystem_respiration(Ta, Rref, Q10=1.5, Tref=10):
    """
    Calculate ecosystem respiration using the Q10 model.

    Parameters:
    - Ta (float or array): Ambient temperature (degC).
    - Rref (float): Reference respiration rate at the reference temperature (Tref) [e.g., umol CO2 m-2 s-1].
    - Q10 (float, optional): Temperature sensitivity factor (default is 1.5).
    - Tref (float, optional): Reference temperature (degC, default is 10).

    Returns:
    - float or array: Ecosystem respiration (Reco) at temperature Ta.
    """
    B = np.log(Q10) / 10  # Temperature sensitivity scaling factor
    Reco = Rref * np.exp(B * (Ta - Tref))  # Ecosystem respiration
    return Reco