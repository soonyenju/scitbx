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