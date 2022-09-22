import math
import scipy

def es_calc(airtemp):
    '''
    Function to calculate saturated vapour pressure from temperature.

    For T<0 C the saturation vapour pressure equation for ice is used
    accoring to Goff and Gratch (1946), whereas for T>=0 C that of
    Goff (1957) is used.
    
    Parameters:
        - airtemp : (data-type) measured air temperature [Celsius].
        
    Returns:
        - es : (data-type) saturated vapour pressure [Pa].

    References
    ----------
    
    - Goff, J.A.,and S. Gratch, Low-pressure properties of water from -160 \
    to 212 F. Transactions of the American society of heating and \
    ventilating engineers, p. 95-122, presented at the 52nd annual \
    meeting of the American society of \
    heating and ventilating engineers, New York, 1946.
    - Goff, J. A. Saturation pressure of water on the new Kelvin \
    temperature scale, Transactions of the American \
    society of heating and ventilating engineers, pp 347-354, \
    presented at the semi-annual meeting of the American \
    society of heating and ventilating engineers, Murray Bay, \
    Quebec. Canada, 1957.

    Examples
    --------    
        >>> es_calc(30.0)
        4242.725994656632
        >>> x = [20, 25]
        >>> es_calc(x)
        array([ 2337.08019792,  3166.82441912])
    
    '''

    # Determine length of array
    n = np.size(airtemp)
    # Check if we have a single (array) value or an array
    if n < 2:
        # Calculate saturated vapour pressures, distinguish between water/ice
        if airtemp < 0:
            # Calculate saturation vapour pressure for ice
            log_pi = - 9.09718 * (273.16 / (airtemp + 273.15) - 1.0) \
                     - 3.56654 * math.log10(273.16 / (airtemp + 273.15)) \
                     + 0.876793 * (1.0 - (airtemp + 273.15) / 273.16) \
                     + math.log10(6.1071)
            es = math.pow(10, log_pi)   
        else:
            # Calculate saturation vapour pressure for water
            log_pw = 10.79574 * (1.0 - 273.16 / (airtemp + 273.15)) \
                     - 5.02800 * math.log10((airtemp + 273.15) / 273.16) \
                     + 1.50475E-4 * (1 - math.pow(10, (-8.2969 * ((airtemp +\
                     273.15) / 273.16 - 1.0)))) + 0.42873E-3 * \
                     (math.pow(10, (+4.76955 * (1.0 - 273.16\
                     / (airtemp + 273.15)))) - 1) + 0.78614
            es = math.pow(10, log_pw)
    else:   # Dealing with an array     
        # Initiate the output array
        es = scipy.zeros(n)
        # Calculate saturated vapour pressures, distinguish between water/ice
        for i in range(0, n):              
            if airtemp[i] < 0:
                # Saturation vapour pressure equation for ice
                log_pi = - 9.09718 * (273.16 / (airtemp[i] + 273.15) - 1.0) \
                         - 3.56654 * math.log10(273.16 / (airtemp[i] + 273.15)) \
                         + 0.876793 * (1.0 - (airtemp[i] + 273.15) / 273.16) \
                         + math.log10(6.1071)
                es[i] = math.pow(10, log_pi)
            else:
                # Calculate saturation vapour pressure for water  
                log_pw = 10.79574 * (1.0 - 273.16 / (airtemp[i] + 273.15)) \
                         - 5.02800 * math.log10((airtemp[i] + 273.15) / 273.16) \
                         + 1.50475E-4 * (1 - math.pow(10, (-8.2969\
                         * ((airtemp[i] + 273.15) / 273.16 - 1.0)))) + 0.42873E-3\
                         * (math.pow(10, (+4.76955 * (1.0 - 273.16\
                         / (airtemp[i] + 273.15)))) - 1) + 0.78614
                es[i] = pow(10, log_pw)
    # Convert from hPa to Pa
    es = es * 100.0
    return es # in Pa

def ea_calc(airtemp, rh):
    '''
    Function to calculate actual saturation vapour pressure.

    Parameters:
        - airtemp: array of measured air temperatures [Celsius].
        - rh: Relative humidity [%].

    Returns:
        - ea: array of actual vapour pressure [Pa].

    Examples
    --------
    
        >>> ea_calc(25,60)
        1900.0946514729308

    '''

    # Calculate saturation vapour pressures
    es = es_calc(airtemp)
    # Calculate actual vapour pressure
    eact = rh / 100.0 * es
    return eact # in Pa

def vpd_calc(airtemp, rh):
    '''
    Function to calculate vapour pressure deficit.

    Parameters:
        - airtemp: measured air temperatures [Celsius].
        - rh: (array of) rRelative humidity [%].
        
    Returns:
        - vpd: (array of) vapour pressure deficits [Pa].
        
    Examples
    --------
    
        >>> vpd_calc(30,60)
        1697.090397862653
        >>> T=[20,25]
        >>> RH=[50,100]
        >>> vpd_calc(T,RH)
        array([ 1168.54009896,     0.        ])
        
    '''
    
    
    # Calculate saturation vapour pressures
    es = es_calc(airtemp)
    eact = ea_calc(airtemp, rh) 
    # Calculate vapour pressure deficit
    vpd = es - eact
    return vpd # in hPa

def saturation_vapor_pressure(T, es0 = 6.112):
    """
    Input:
    ------
    T(degC):
        Temperature
    es0 (hPa):
        saturation_vapor_pressure at 0 degC
    Output:
    -------
    saturation_vapor_pressure(hPa)
    math:
    -----
    Tetens formula
    6.112 e^((17.67 * T) / (T + 243.5)
    """
    es = 6.112 * np.e ** ((17.67 * T) / (T + 243.5))
    return es