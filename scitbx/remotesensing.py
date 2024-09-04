def get_NDVI(r, nir):
    ndvi = (nir - r) / (nir + r)
    return ndvi

def get_NIRv(ndvi, nir):
    nirv = ndvi * nir
    return nirv

def get_kNDVI(ndvi):
    kndvi = np.tanh(ndvi**2)
    return kndvi

def get_EVI2band(r, nir):
    evi = 2.5 * (nir - r) / (nir + 2.4 * r + 1)
    return evi


def get_MODIS_IGBPcode():
    '''
    url: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1 
    '''
    MODIS_IGBP_codes = ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'CSH', 'OSH', 'WSA', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CNV', 'SNO', 'BSV', 'WAB']
    MODIS_IGBP_dict = dict(zip(MODIS_IGBP_codes, np.arange(len(MODIS_IGBP_codes)) + 1))
    return MODIS_IGBP_dict

def convert_gCm2d1_PgCyr_025deg():
    '''
    Terrestrial ecosystem carbon flux unit conversion:
    from gC m-2 d-1 to Pg C yr-1 for spatial resolution of 0.25 deg
    '''
    coef = 365 * 0.25 * 0.25 * 1e5 * 1e5 / 1e15
    return coef