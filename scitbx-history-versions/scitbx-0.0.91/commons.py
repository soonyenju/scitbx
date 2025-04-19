ec_unit_DD = {
    'GPP': '$gC \, m^{-2} \, d^{-1}$',
    'Reco': '$gC \, m^{-2} \, d^{-1}$',
    'NEE': '$gC \, m^{-2} \, d^{-1}$',
    'H': '$W \, m^{-2}$',
    'LE': '$W \, m^{-2}$',
}

ec_unit_HH = {
    'GPP': '$\mu molCO_2 \, m^{-2} \, s^{-1}$',
    'Reco': '$\mu molCO_2 \, m^{-2} \, s^{-1}$',
    'NEE': '$\mu molCO_2 \, m^{-2} \, s^{-1}$',
    'H': '$W \, m^{-2}$',
    'LE': '$W \, m^{-2}$',
}

coef_DD_SS = 86400 # 1 day => seconds

coef_MgC_gC = 1e6 # MgC => gC
coef_PgC_gC = 1e15 # PgC => gC
coef_TgC_gC = 1e12 # TgC => gC

coef_umm2s1_gCm2d1 = 1.03772448 # um m-2 s-1 => gC m-2 d-1
coef_ha_m2 = 1e4 # ha => m-2
coef_Wm2_mmyr1 = 14 # W m-2 => mm yr-1 for evapotranspiration

molar_mass_CO2 = 44.009 # g/mol
molar_mass_H2O = 18.01528 # g/mol