import numpy as np

'''
Source: https://github.com/ImperialCollegeLondon/pyrealm
'''

def calc_co2_to_ca(co2, patm):
    r"""CO2 ppm to Pa"""

    return 1.0e-6 * co2 * patm  # Pa, atms. CO2

def calc_ftemp_arrh(tk, ha):
    """Calculate enzyme kinetics scaling factor."""
    # Standard baseline reference temperature of photosynthetic processes (Prentice,
    # unpublished)  (:math:`T_o` , 25.0, °C)
    plant_T_ref = 25.0
    # Conversion from °C to K   (:math:`CtoK` , 273.15, -)
    k_CtoK = 273.15
    # Universal gas constant (:math:`R` , 8.3145, J/mol/K)
    k_R = 8.3145

    tkref = plant_T_ref + k_CtoK

    return np.exp(ha * (tk - tkref) / (tkref * k_R * tk))


def calc_gammastar(tc, patm):
    # Calculate the photorespiratory CO2 compensation point.
    # Bernacchi estimate of gs25_0
    bernacchi_gs25_0 = 4.332  # Reported as 42.75 µmol mol-1
    # Standard reference atmosphere (Allen, 1973) (:math:`P_o` , 101325.0, Pa)
    k_Po = 101325.0
    # Conversion from °C to K   (:math:`CtoK` , 273.15, -)
    k_CtoK = 273.15
    # Bernacchi estimate of activation energy for gammastar (J/mol)
    bernacchi_dha = 37830
    return (
        bernacchi_gs25_0 * patm / k_Po
        * calc_ftemp_arrh((tc + k_CtoK), ha = bernacchi_dha)
    )

def evaluate_horner_polynomial(x, cf):
    """Evaluates a polynomial with coefficients `cf` at `x` using Horner's method."""
    y = np.zeros_like(x)
    for c in reversed(cf):
        y = x * y + c
    return y

def calc_density_h2o_fisher(tc, patm):
    """Calculate water density."""

    # Calculate lambda, (bar cm^3)/g:
    # Fisher Dial
    fisher_dial_lambda = np.array([1788.316, 21.55053, -0.4695911, 0.003096363, -7.341182e-06])

    lambda_coef = fisher_dial_lambda
    lambda_val = evaluate_horner_polynomial(tc, lambda_coef)

    # Calculate po, bar
    fisher_dial_Po = np.array([5918.499, 58.05267, -1.1253317, 0.0066123869, -1.4661625e-05])
    po_coef = fisher_dial_Po
    po_val = evaluate_horner_polynomial(tc, po_coef)

    # Calculate vinf, cm^3/g

    fisher_dial_Vinf = np.array([
        0.6980547, -0.0007435626, 3.704258e-05, -6.315724e-07, 9.829576e-09,
        -1.197269e-10, 1.005461e-12, -5.437898e-15, 1.69946e-17, -2.295063e-20
    ])

    vinf_coef = fisher_dial_Vinf
    vinf_val = evaluate_horner_polynomial(tc, vinf_coef)

    # Convert pressure to bars (1 bar <- 100000 Pa)
    pbar = 1e-5 * patm

    # Calculate the specific volume (cm^3 g^-1):
    spec_vol = vinf_val + lambda_val / (po_val + pbar)

    # Convert to density (g cm^-3) -> 1000 g/kg; 1000000 cm^3/m^3 -> kg/m^3:
    rho = 1e3 / spec_vol
    return rho

def calc_density_h2o_chen(tc, p):
    """Calculate the density of water using Chen et al 2008."""

    # Calculate density at 1 atm (kg/m^3):
    # Chen water density
    chen_po = np.array([
        0.99983952, 6.788260e-5, -9.08659e-6, 1.022130e-7, -1.35439e-9,
        1.471150e-11, -1.11663e-13, 5.044070e-16, -1.00659e-18,
    ])
    po_coef = chen_po
    po = evaluate_horner_polynomial(tc, po_coef)

    # Calculate bulk modulus at 1 atm (bar):
    chen_ko = np.array([19652.17, 148.1830, -2.29995, 0.01281, -4.91564e-5, 1.035530e-7])
    ko_coef = chen_ko
    ko = evaluate_horner_polynomial(tc, ko_coef)

    # Calculate temperature dependent coefficients:
    chen_ca = np.array([3.26138, 5.223e-4, 1.324e-4, -7.655e-7, 8.584e-10])
    ca_coef = chen_ca
    ca = evaluate_horner_polynomial(tc, ca_coef)

    chen_cb = np.array([7.2061e-5, -5.8948e-6, 8.69900e-8, -1.0100e-9, 4.3220e-12])
    cb_coef = chen_cb
    cb = evaluate_horner_polynomial(tc, cb_coef)

    # Convert atmospheric pressure to bar (1 bar = 100000 Pa)
    pbar = (1.0e-5) * p

    pw = ko + ca * pbar + cb * pbar**2.0
    pw /= ko + ca * pbar + cb * pbar**2.0 - pbar
    pw *= (1e3) * po
    return pw

def calc_density_h2o(tc, patm, safe = True, water_density_method = 'fisher'):
    """Calculate water density."""

    # Safe guard against instability in functions at low temperature.
    if safe and np.nanmin(tc) < np.array([-30]):
        raise ValueError(
            "Water density calculations below about -30°C are "
            "unstable. See argument safe to calc_density_h2o"
        )

    if water_density_method == "fisher":
        return calc_density_h2o_fisher(tc, patm)

    if water_density_method == "chen":
        return calc_density_h2o_chen(tc, patm)

    raise ValueError("Unknown method provided to calc_density_h2o")

def calc_viscosity_h2o(tc, patm, simple = False, simple_viscosity = False):
    """Calculate the viscosity of water."""

    if simple or simple_viscosity:
        # The reference for this is unknown, but is used in some implementations
        # so is included here to allow intercomparison.
        return np.exp(-3.719 + 580 / ((tc + 273) - 138))

    # Get the density of water, kg/m^3
    rho = calc_density_h2o(tc, patm)

    # Calculate dimensionless parameters:
    # Conversion from °C to K   (:math:`CtoK` , 273.15, -)
    k_CtoK = 273.15
    # Huber reference temperature (:math:`tk_{ast}`, 647.096, Kelvin)
    huber_tk_ast = 647.096
    # Huber reference density (:math:`\rho_{ast}`, 322.0, kg/m^3)
    huber_rho_ast = 322.0
    tbar = (tc + k_CtoK) / huber_tk_ast
    rbar = rho / huber_rho_ast

    # Calculate mu0 (Eq. 11 & Table 2, Huber et al., 2009):
    # Temperature dependent parameterisation of Hi in Huber.
    huber_H_i = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
    mu0 = huber_H_i[0] + huber_H_i[1] / tbar
    mu0 += huber_H_i[2] / (tbar * tbar)
    mu0 += huber_H_i[3] / (tbar * tbar * tbar)
    mu0 = (1e2 * np.sqrt(tbar)) / mu0

    # Calculate mu1 (Eq. 12 & Table 3, Huber et al., 2009):
    ctbar = (1.0 / tbar) - 1.0
    mu1 = 0.0

    # Iterate over the rows of the H_ij core_constants matrix
    # Temperature and mass density dependent parameterisation of Hij in Huber.
    huber_H_ij = np.array([
                [0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0],
                [0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573],
                [-0.281378, -0.906851, -0.772479, -0.489837, -0.25704, 0.0],
                [0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0],
                [-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0],
                [0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264],
            ])

    for row_idx in np.arange(huber_H_ij.shape[1]):
        cf1 = ctbar**row_idx
        cf2 = 0.0
        for col_idx in np.arange(huber_H_ij.shape[0]):
            cf2 += huber_H_ij[col_idx, row_idx] * (rbar - 1.0) ** col_idx
        mu1 += cf1 * cf2

    mu1 = np.exp(rbar * mu1)

    # Calculate mu_bar (Eq. 2, Huber et al., 2009), assumes mu2 = 1
    mu_bar = mu0 * mu1

    # Calculate mu (Eq. 1, Huber et al., 2009)
    # Huber reference pressure (:math:`\mu_{ast}` 1.0e-6, Pa s)
    huber_mu_ast = 1e-06
    return mu_bar * huber_mu_ast  # Pa s

def calc_ns_star(tc, patm):
    r"""Calculate the relative viscosity of water."""

    visc_env = calc_viscosity_h2o(tc, patm)
    # Standard reference temperature (:math:`T_o` ,  298.15, K)
    k_To = 298.15
    # Conversion from °C to K   (:math:`CtoK` , 273.15, -)
    k_CtoK = 273.15
    # Standard reference atmosphere (Allen, 1973) (:math:`P_o` , 101325.0, Pa)
    k_Po = 101325.0
    visc_std = calc_viscosity_h2o(np.array(k_To) - np.array(k_CtoK), np.array(k_Po))

    return visc_env / visc_std

def calc_kmm(tc, patm):
    """Calculate the Michaelis Menten coefficient of Rubisco-limited assimilation."""

    # conversion to Kelvin
    # Conversion from °C to K   (:math:`CtoK` , 273.15, -)
    k_CtoK = 273.15
    tk = tc + k_CtoK

    # Bernacchi estimate of kc25
    bernacchi_kc25 = 39.97  # Reported as 404.9 µmol mol-1
    # Bernacchi estimate of activation energy Kc for CO2 (J/mol)
    bernacchi_dhac = 79430
    kc = bernacchi_kc25 * calc_ftemp_arrh(
        tk, ha = bernacchi_dhac
    )
    # Bernacchi estimate of ko25
    bernacchi_ko25 = 27480  # Reported as 278.4 mmol mol-1
    # Bernacchi estimate of activation energy Ko for O2 (J/mol)
    bernacchi_dhao = 36380
    ko = bernacchi_ko25 * calc_ftemp_arrh(
        tk, ha = bernacchi_dhao
    )

    # O2 partial pressure
    # O2 partial pressure, Standard Atmosphere (:math:`co` , 209476.0, ppm)
    k_co = 209476.0
    po = k_co * 1e-6 * patm

    return kc * (1.0 + po / ko)


def wang17(mj):
    """Calculate limitation factors following :cite:`Wang:2017go`."""
    # Unit carbon cost for the maintenance of electron transport capacity (:math:`c`, 0.41, )
    wang17_c = 0.41

    vals_defined = np.greater(mj, wang17_c)

    f_v = np.sqrt(1 - (wang17_c / mj) ** (2.0 / 3.0), where=vals_defined)
    f_j = np.sqrt((mj / wang17_c) ** (2.0 / 3.0) - 1, where=vals_defined)

    # Backfill undefined values - tackling float vs np.ndarray types
    if isinstance(f_v, np.ndarray):
        f_j[np.logical_not(vals_defined)] = np.nan  # type: ignore
        f_v[np.logical_not(vals_defined)] = np.nan  # type: ignore
    elif not vals_defined:
        f_j = np.nan
        f_v = np.nan
    return f_j, f_v

def smith19(mj):
    """Calculate limitation factors following :cite:`Smith:2019dv`."""

    # Adopted from Nick Smith's code:
    # Calculate omega, see Smith et al., 2019 Ecology Letters  # Eq. S4
    # Scaling factor theta for Jmax limitation (:math:`\theta`, 0.85)
    smith19_theta = 0.85
    theta = smith19_theta
    # Scaling factor c for Jmax limitation (:math:`c`, 0.05336251)
    smith19_c_cost = 0.05336251
    c_cost = smith19_c_cost

    # simplification terms for omega calculation
    cm = 4 * c_cost / mj
    v = 1 / (cm * (1 - smith19_theta * cm)) - 4 * theta

    # account for non-linearities at low m values. This code finds
    # the roots of a quadratic function that is defined purely from
    # the scalar theta, so will always be a scalar. The first root
    # is then used to set a filter for calculating omega.

    cap_p = (((1 / 1.4) - 0.7) ** 2 / (1 - theta)) + 3.4
    aquad = -1
    bquad = cap_p
    cquad = -(cap_p * theta)
    roots = np.polynomial.polynomial.polyroots(
        [aquad, bquad, cquad]
    )  # type: ignore [no-untyped-call]

    # factors derived as in Smith et al., 2019
    m_star = (4 * c_cost) / roots[0].real
    omega = np.where(
        mj < m_star,
        -(1 - (2 * theta)) - np.sqrt((1 - theta) * v),
        -(1 - (2 * theta)) + np.sqrt((1 - theta) * v),
    )

    # np.where _always_ returns an array, so catch scalars
    omega = omega.item() if np.ndim(omega) == 0 else omega

    omega_star = (
        1.0
        + omega
        - np.sqrt((1.0 + omega) ** 2 - (4.0 * theta * omega))  # Eq. 18
    )

    # Effect of Jmax limitation - note scaling here. Smith et al use
    # phi0 as as the quantum efficiency of electron transport, which is
    # 4 times our definition of phio0 as the quantum efficiency of photosynthesis.
    # So omega*/8 theta and omega / 4 are scaled down here  by a factor of 4.
    # Ignore `mypy` here as omega_star is explicitly not None.
    f_v = omega_star / (2.0 * theta)  # type: ignore
    f_j = omega
    return f_j, f_v

def simple():
    """Apply the 'simple' form of the equations."""

    # Set Jmax limitation to unity - could define as 1.0 in __init__ and
    # pass here, but setting explicitly within the method for clarity.
    f_v = np.array([1.0])
    f_j = np.array([1.0])
    return f_j, f_v

def calc_ftemp_kphio(tc, c4 = False):
    # Quadratic scaling of Kphio with temperature
    kphio_C4 = (-0.064, 0.03, -0.000464)
    kphio_C3 = (0.352, 0.022, -0.00034)

    if c4:
        coef = kphio_C4
    else:
        coef = kphio_C3

    ftemp = coef[0] + coef[1] * tc + coef[2] * tc**2
    ftemp = np.clip(ftemp, 0.0, None)

    return ftemp


def calc_light_water_use_efficiency(tc, patm, ca, vpd, do_ftemp_kphio, c4 = False, limitation_factors = 'wang17'):
    """
    The basic calculation of LUE = phi0 * M_c * m with an added penalty term for jmax limitation
    """
    k_c_molmass = 12.0107
    # Molecular mass of carbon (:math:`c_molmass` , 12.0107, g)

    # Set context specific defaults for kphio to match Stocker paper
    if not do_ftemp_kphio:
        init_kphio = 0.049977
    else:
        init_kphio = 0.081785

    if do_ftemp_kphio:
        ftemp_kphio = calc_ftemp_kphio(tc, c4 = c4)
        kphio = init_kphio * ftemp_kphio
    else:
        kphio = np.array([init_kphio])

    gammastar = calc_gammastar(tc, patm)
    ns_star = calc_ns_star(tc, patm)
    kmm = calc_kmm(tc, patm)

    if c4:
        # Unit cost ratio for C4 plants (:math:`\beta`, 16.222).
        beta_cost_ratio_c4 = np.array([146.0 / 9])
        beta = beta_cost_ratio_c4

    else:
        beta_cost_ratio_prentice14 = np.array([146.0])
        beta = beta_cost_ratio_prentice14

    xi = np.sqrt((beta * (kmm + gammastar)) / (1.6 * ns_star))

    chi = gammastar / ca + (1.0 - gammastar / ca) * xi / (xi + np.sqrt(vpd))

    ci = chi * ca
    if c4:
        mj = 1
    else:
        mj = (ci - gammastar) / (ci + 2 * gammastar)

    if limitation_factors == 'wang17':
        f_j, f_v = wang17(mj)
    elif limitation_factors == 'smith19':
        f_j, f_v = smith19(mj)
    else:
        f_j, f_v = simple()

    lue = kphio * mj * f_v * k_c_molmass
    # Intrinsic water use efficiency (iWUE, µmol mol-1)
    iwue = (5 / 8 * (ca - ci)) / (1e-6 * patm)

    return lue, iwue

'''
# Example
# fapar: the fraction of absorbed photosynthetically active radiation (-)
# ppfd: photosynthetic photon flux density (µmol m-2 s-1)
# iabs = fapar * ppfd

tc = 20
patm = 101325.0
co2 = 400
vpd = 100
do_ftemp_kphio = True

ca = calc_co2_to_ca(co2, patm)
# # C3:
print(
    calc_light_water_use_efficiency(tc, patm, ca, vpd, do_ftemp_kphio, c4 = False, limitation_factors = 'wang17')
)
# # C4:
print(
    calc_light_water_use_efficiency(tc, patm, ca, vpd, do_ftemp_kphio, c4 = True, limitation_factors = 'none')
)

# -----------------------------
# Check against

# !pip install pyrealm

from pyrealm import pmodel

env  = pmodel.PModelEnvironment(tc=tc, patm=patm, vpd=vpd, co2=co2)
# # C3
print(
    pmodel.PModel(env, method_optchi='prentice14').lue
)
# # C4
print(
    pmodel.PModel(env, method_optchi='c4', method_jmaxlim='none').lue
)

# mod_c4 = PModel(env, method_optchi='c4', method_jmaxlim='none')
'''

# env  = pmodel.PModelEnvironment(tc=dfp['TA_F_MDS'].values, patm=dfp['PA'].values * 100, vpd=dfp['VPD_F_MDS'].values * 100, co2=dfp['CO2_F_MDS'].values)
# model = pmodel.PModel(env, method_optchi=c3c4) # or c4
# model.estimate_productivity(fapar=dfp['NIRv'].values * 0.5, ppfd=dfp['PPFD_IN'].values)