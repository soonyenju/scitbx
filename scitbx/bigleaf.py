'''
Python code is translated from R version: https://cran.r-project.org/web/packages/bigleaf/bigleaf.pdf
'''

def intercellular_CO2(Ca, GPP, Gs, Rleaf = 0):
    '''
    Ca: Atmospheric or surface CO2 concentration (umol mol-1)
    GPP: Gross primary productivity (umol CO2 m-2 s-1)
    Gs: Surface conductance to water vapor (mol m-2 s-1)
    Rleaf: Ecosystem respiration stemming from leaves (umol CO2 m-2 s-1); defaults to 0
    '''
    Ci = Ca - (GPP - Rleaf) / (Gs / 1.6)
    return Ci

def Rg_to_PPFD(Rg, J_to_mol = 4.6, frac_PAR = 0.5):
    '''
    Rg: Global radiation = incoming short-wave radiation at the surface (W m-2)
    J_to_mol: Conversion factor from J m-2 s-1 (= W m-2) to umol (quanta) m-2 s-1
    frac_PAR: Fraction of incoming solar irradiance that is photosynthetically active radiation (PAR); defaults to 0.5
    PPFD: Photosynthetic photon flux density (umol m-2 s-1)
    '''
    PPFD = Rg * frac_PAR * J_to_mol
    return PPFD

def light_response(NEE, PPFD, Reco):
    from scipy.optimize import curve_fit
    '''
    mod <- nls(-NEE ~ alpha * PPFD / (1 - (PPFD / PPFD_ref) + (alpha * PPFD / GPP_ref)) - Reco,
                start=list(alpha=0.05,GPP_ref=30),...)
    NEE: umol CO2 m-2 s-1
    Reco: umol CO2 m-2 s-1
    PPFD: umol m-2 s-1
    PPFD_ref = 2000 umol m-2 s-1; Falge et al. 2001
    '''
    X = [PPFD, Reco]
    def func(X, alpha, GPP_ref, PPFD_ref = 2000):
        PPFD = X[0]
        Reco = X[1]
        return alpha * PPFD / (1 - (PPFD / PPFD_ref) + (alpha * PPFD / GPP_ref)) - Reco
    popt, pcov = curve_fit(func,  X,  -NEE) 
    return popt, pcov

def latent_heat_vaporization(Tair):
    '''
    Tair Air temperature (deg C)
    λ is Latent heat of vaporization (J kg-1)
    '''
    lambda_ = (2.501 - 0.00237 * Tair) * 10e6
    return lambda_

def LE_to_ET(LE, Tair):
    '''
    LE: Latent heat flux (W m-2)
    Tair: Air temperature (deg C)
    ET: Evapotranspiration (kg m-2 s-1)
    where λ is the latent heat of vaporization (J kg-1) as calculated by latent_heat_vaporization
    '''
    lambda_ = latent_heat_vaporization(Tair)
    ET = LE / lambda_
    return ET

def ET_to_LE(ET, Tair):
    '''
    LE: Latent heat flux (W m-2)
    Tair: Air temperature (deg C)
    ET: Evapotranspiration (kg m-2 s-1)
    where λ is the latent heat of vaporization (J kg-1) as calculated by latent_heat_vaporization
    '''
    lambda_ = latent_heat_vaporization(Tair)
    LE = ET * lambda_
    return LE