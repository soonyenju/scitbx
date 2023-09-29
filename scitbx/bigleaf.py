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
