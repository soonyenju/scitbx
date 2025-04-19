import numpy as np

# Priestley_Taylor can be used for forests and grasslands for approximation, but with different scaling factor alpha
def Priestley_Taylor(Ta, PA, R, G = 0, alpha = 1.26):
    '''
    Potential evapotranspiration [mm d-1] priestley_assessment_1972
    ETo = alpha * (R-G) * delta_ / (lambda_ * (delta_ + gamma_))
    https://wetlandscapes.github.io/blog/blog/penman-monteith-and-priestley-taylor/

    where is the dryness coefficient, R is the net incoming radiation, G is the heat flux into the ground (R - G =
    LE + H) where H is sensible heat and LE is latent heat, delta_ is the slope of the saturation vapour pressure
    curve, and gamma_ is the psychrometric constant.

    Ta: average day temperature [degC].
    PA: atmospheric pressure [kPa].
    R: net radiation [MJ m-2 d-1]. W m-2 * (1000000/86400) = MJ m-2 d-1
    G: soil heat flux [MJ m-2 d-1].
    '''
    # Latent Heat of Vaporization [MJ kg-1].
    lambda_ = 2.501 - 0.002361 * Ta
    # Specific heat of air [MJ kg-1 degC-1]
    CP = 1.013 * 10**-3
    # Psychrometric constant [kPa degC-1].
    gamma_ = CP * PA / (0.622 * lambda_)

    # Saturation vapor pressure at the air temperature T [kPa]
    es = 0.6108 * np.exp(17.27 * Ta / (Ta + 237.3))
    # Slope of saturation vapour pressure curve at air Temperature [kPa °C-1].
    delta_ = 4098 * es / (Ta + 237.3) ** 2

    ETo = (alpha * delta_ * (R - G)) / (lambda_ * (delta_ + gamma_))
    return ETo


def Priestley_Taylor_JPL(net_radiation, Ta, NDVI):
    """
    Calculate evapotranspiration (ET) using the PT-JPL model.

    Parameters:
    - net_radiation: Net radiation (W / m2)
    - albedo: Surface albedo
    - Ta: Air temperature (degC)
    - NDVI: Normalised Difference Vegetation Index

    Returns:
    - et: Evapotranspiration (mm/day)
    Citation:
    - Fisher, Joshua B., Kevin P. Tu, and Dennis D. Baldocchi. "Global estimates of the land-atmosphere water flux based on monthly AVHRR and ISLSCP-II data, validated at 16 FLUXNET sites." Remote Sensing of Environment 112.3 (2008): 901-919.
    - Wen, Jiaming, et al. "Resolve the clear-sky continuous diurnal cycle of high-resolution ECOSTRESS evapotranspiration and land surface temperature." Water Resources Research 58.9 (2022): e2022WR032227.
    """
    Ta = Ta + 273.15
    # Constants
    lambda_v = 2.45e6  # Latent heat of vaporisation (J/kg)
    gamma = 0.066  # Psychrometric constant (kPa/degC)


    # Vapour pressure deficit (VPD, kPa)
    es = 0.6108 * np.exp((17.27 * (Ta - 273.15)) / ((Ta - 273.15) + 237.3))  # Saturation vapour pressure (kPa)

    # Slope of saturation vapour pressure curve (Delta, kPa/degC)
    delta = 4098 * es / ((Ta - 273.15) + 237.3) ** 2

    # Vegetation constraints (NDVI-based f_veg)
    f_veg = np.clip((NDVI - 0.2) / 0.8, 0, 1)  # Scaling NDVI to [0, 1]


    # Alpha (Priestley-Taylor coefficient)
    alpha = 1.26 * f_veg

    # ET calculation (mm/day)
    et = alpha * delta * net_radiation / (delta + gamma) / lambda_v  # kg/m2/s (convert to mm/day)
    et = et * 86400  # Convert from kg/m2/s to mm/day

    return np.maximum(et, 0)  # Ensure ET is non-negative


def mmday2Wm2(et_mm_day):
    """
    Conversion of mm/day to W/m2:
    1 mm/day = 1 kg/m2/day (since 1 mm of water depth equals 1 kg/m2)
    Latent heat of vaporisation (L_v) = ca. 2.45 MJ/kg = 2.45 x 10^6 J/kg
    Seconds in a day = 24 x 60 x 60 = 86400 s
    Energy flux (W/m2) = (L_v x evaporation rate) / seconds per day
    For 1 mm/day: (2.45 x 10^6 J/kg x 1 kg/m2/day) / 86400 s = ca. 28.4 W/m2

    1 mm/day = 28.4 W/m2
    """
    et_Wm2 = et_mm_day * 28.4
    return et_Wm2

