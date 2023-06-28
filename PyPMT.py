#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from random import choice, seed
import math
from math import isnan
import csv
import pyproj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


# In[ ]:


def load_csv_data(filename):
    lat = []
    lon = []
    names = []

    with open(filename, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row

        for row in csvreader:
            try:
                lat_val = float(row[5])
                lon_val = float(row[6])
                lat.append(lat_val)
                lon.append(lon_val)
                names.append(row[1])
            except ValueError:
                continue

    return np.array(lat), np.array(lon), np.array(names)


def strlookup(featurename, names):
    featurename = featurename.lower()  # Convert featurename to lowercase for case-insensitive matching

    indices = [i for i, name in enumerate(names) if featurename in name.lower()]

    if len(indices) > 0:
        x = indices[0]  # Choose the first matching index
        NearbyNames = [names[i] for i in indices]  # Get all matching names
    else:
        x = None
        NearbyNames = []

    return x, NearbyNames


def scarloc(featurename, *varargin):
    
    # Check if featurename is a list or a single string
    if isinstance(featurename, str):
        featurename = [featurename]

    # Handle optional arguments
    OfferHelp = False
    pscoords = False
    kmout = False

    if len(varargin) > 0:
        if isinstance(varargin[0], bool):
            OfferHelp = varargin[0]
        if len(varargin) >= 1:
            if 'xy' in varargin[0]:
                pscoords = True
                if len(varargin) >= 2 and 'km' in varargin[1]:
                    kmout = True

    # Load data from CSV file
    lat, lon, names = load_csv_data('SCAR_CGA_PLACE_NAMES.csv')

    featurelat = np.full(len(featurename), np.nan)
    featurelon = np.full(len(featurename), np.nan)

    # Look for each feature name
    for k in range(len(featurelat)):
        x, NearbyNames = strlookup(featurename[k], names)
        if x is None and OfferHelp:
            fmsg = [
                f'"{featurename[k]}" not found.',
                f'Are you sure that "{featurename[k]}" exists in Antarctica?',
                'Did a cat walk across your keyboard?',
                'This is the real reason one shouldn''t text and drive. Check your spelling and try again.',
                'Now you''re just making things up.',
                f'SCAR has identified more than 25,000 features in Antarctica, but "{featurename[k]}" is not one of them.',
                f'Can''t find "{featurename[k]}".',
                f'"{featurename[k]}" may exist somewhere in the world, but you won''t find it in Antarctica.',
                f'It is possible that Robert F. Scott named something in Antarctica "{featurename[k]}", but if he did there are no records of it.',
                f'You must be thinking of {featurename[k]}, Kansas, because {featurename[k]}, Antarctica does not exist.',
                f'Sure, they used to just call it {featurename[k]}, but not anymore, what with political correctness and all.',
                f'"{featurename[k]}" is an interesting combination of letters, but I don''t think it''s any place in Antarctica.',
                f'The great Wayne Cochran once sang, "Where oh where can my {featurename[k]} be?" Because it''s not in Antarctica.',
                f'I''m pretty sure it is in violation of the Antarctic Treaty to refer to any place as "{featurename[k]}".',
                f'"{featurename[k]}" does not match any entries in the SCAR database.',
                f'Science is all about formality, so the bigwigs will surely look down their noses at such colloquial jargon as "{featurename[k]}".',
                f'My doctor said I need to get my {featurename[k]} removed.',
                'Frostbitten Antarctic researcher mistypes again.',
                'This may be an issue of American English versus British English.',
                f'Antarctica''s a strange place, but it''s not science fiction. Verify that "{featurename[k]}" actually exists.',
                f'What''s in a name? I''ll tell you what''s in a name: That which you call "{featurename[k]}" by any other name may actually exist in Antarctica.',
                f'Did John Carpenter tell you''ll find "{featurename[k]}" in Antarctica?',
                f'You know, some folks say glaciology is a shrinking field, but I say things are just heating up. In other news, "{featurename[k]}" does not exist.',
                f'You''re a glaciologist? Isn''t that a slow-moving field? Also, I have to tell you, I can''t seem to find any record of "{featurename[k]}".',
                f'Amazing glaciology, how sweet the sound... "{featurename[k]}" once was lost, and still has not been found.'
            ]

            np.random.shuffle(fmsg)
            print(fmsg[0])
            if NearbyNames:
                print('Here are the best matches I can find:')
                print(NearbyNames)
            else:
                print('Try typing "load scarnames" to explore the available list of features.')
            return

        if x is not None:
            featurelat[k] = lat[x]
            featurelon[k] = lon[x]
    
    # Convert to polar stereographic coordinates
    if pscoords:
        featurelat, featurelon = ll2ps(featurelat, featurelon)

    # Convert to polar stereographic kilometers
    if kmout:
        featurelon = featurelon / 1000
        featurelat = featurelat / 1000

    # Returning only latitude or only x would not make any sense,
    # so if no outputs are requested, or if only one output is requested,
    # return as a lat column and lon column or [x y]
    if len(featurename) == 1:
        varargout = np.column_stack((featurelat, featurelon))
        return varargout[0]
    else:
        varargout = [featurelat, featurelon]
        return varargout


# In[ ]:


def strlookup(featurename, names):
    featurename = featurename.lower()  # Convert featurename to lowercase for case-insensitive matching

    indices = [i for i, name in enumerate(names) if featurename in name.lower()]

    if len(indices) > 0:
        x = indices[0]  # Choose the first matching index
        NearbyNames = [names[i] for i in indices]  # Get all matching names
    else:
        x = None
        NearbyNames = []

    return x, NearbyNames


# In[ ]:


def handle_missing_feature(featurename, NearbyNames):
    fmsg = [
        f'"{featurename}" not found.',
        f'Are you sure that "{featurename}" exists in Antarctica?',
        'Did a cat walk across your keyboard?',
        'This is the real reason one shouldn\'t text and drive. Check your spelling and try again.',
        'Now you\'re just making things up.',
        f'SCAR has identified more than 25,000 features in Antarctica, but "{featurename}" is not one of them.',
        f'Can\'t find "{featurename}".',
        f'"{featurename}" may exist somewhere in the world, but you won\'t find it in Antarctica.',
        f'It is possible that Robert F. Scott named something in Antarctica "{featurename}", but if he did there are no records of it.',
        f'You must be thinking of {featurename}, Kansas, because {featurename}, Antarctica does not exist.',
        f'Sure, they used to just call it {featurename}, but not anymore, what with political correctness and all.',
        f'"{featurename}" is an interesting combination of letters, but I don\'t think it\'s any place in Antarctica.',
        f'The great Wayne Cochran once sang, "Where oh where can my {featurename} be?" Because it\'s not in Antarctica.',
        f'I\'m pretty sure it is in violation of the Antarctic Treaty to refer to any place as "{featurename}".',
        f'"{featurename}" does not match any entries in the SCAR database.',
        f'Science is all about formality, so the bigwigs will surely look down their noses at such colloquial jargon as "{featurename}".',
        f'My doctor said I need to get my {featurename} removed.',
        'Frostbitten Antarctic researcher mistypes again.',
        'This may be an issue of American English versus British English.',
        f'Antarctica\'s a strange place, but it\'s not science fiction. Verify that "{featurename}" actually exists.',
        f'What\'s in a name? I\'ll tell you what\'s in a name: That which you call "{featurename}" by any other name may actually exist in Antarctica.',
        f'Did John Carpenter tell you\'ll find "{featurename}" in Antarctica?',
        f'You know, some folks say glaciology is a shrinking field, but I say things are just heating up. In other news, "{featurename}" does not exist.',
        f'You\'re a glaciologist? Isn\'t that a slow-moving field? Also, I have to tell you, I can\'t seem to find any record of "{featurename}".',
        f'Amazing glaciology, how sweet the sound... "{featurename}" once was lost, and still has not been found.'
    ]

    rngstart = seed()  # get initial rng setting before changing it temporarily.
    random_msg = choice(fmsg)
    print(random_msg)
    seed(rngstart)  # returns to original rng settings.

    if NearbyNames:
        print('Here are the best matches I can find:')
        print(NearbyNames)
    else:
        print('Try typing "load scarnames" to explore the available list of features.')

    return np.nan, np.nan


# In[ ]:


def ll2ps(lat, lon):
    # Define the projection parameters
    proj_params = {
        'proj': 'stere',
        'lat_0': 90, # suitable for the southern hemisphere
        'lon_0': 0,
        'lat_ts': -71,
        'a': 6378137,
        'b': 6356752.3
    }

    # Create the projection object
    proj = pyproj.Proj(proj_params)

    # Convert latitude and longitude to polar stereographic coordinates
    x, y = proj(lon, lat)

    return x, y


# In[ ]:


def ps2ll(x, y, **kwargs):
    # Define default values for optional keyword arguments
    phi_c = -71  # standard parallel (degrees)
    a = 6378137.0  # radius of ellipsoid, WGS84 (meters)
    e = 0.08181919  # eccentricity, WGS84
    lambda_0 = 0  # meridian along positive Y axis (degrees)

    # Parse optional keyword arguments
    for key, value in kwargs.items():
        if key.lower() == 'true_lat':
            phi_c = value
            assert isinstance(phi_c, (int, float)), 'True lat must be a scalar.'
            if phi_c > 0:
                print("I'm assuming you forgot the negative sign for the true latitude, \
                      and I am converting your northern hemisphere value to southern hemisphere.")
                phi_c = -phi_c
        elif key.lower() == 'earth_radius':
            a = value
            assert isinstance(a, (int, float)), 'Earth radius must be a scalar.'
            assert a > 7e+3, 'Earth radius should be something like 6378137 in meters.'
        elif key.lower() == 'eccentricity':
            e = value
            assert isinstance(e, (int, float)), 'Earth eccentricity must be a scalar.'
            assert 0 <= e < 1, 'Earth eccentricity does not seem like a reasonable value.'
        elif key.lower() == 'meridian':
            lambda_0 = value
            assert isinstance(lambda_0, (int, float)), 'meridian must be a scalar.'
            assert -180 <= lambda_0 <= 360, 'meridian does not seem like a logical value.'
        else:
            print("At least one of your input arguments is invalid. Please try again.")
            return 0

    # Convert to radians and switch signs
    phi_c = -phi_c * np.pi / 180
    lambda_0 = -lambda_0 * np.pi / 180
    x = -x
    y = -y

    # Calculate constants
    t_c = np.tan(np.pi / 4 - phi_c / 2) / ((1 - e * np.sin(phi_c)) / (1 + e * np.sin(phi_c))) ** (e / 2)
    m_c = np.cos(phi_c) / np.sqrt(1 - e ** 2 * (np.sin(phi_c)) ** 2)

    # Calculate rho and t
    rho = np.sqrt(x ** 2 + y ** 2)
    t = rho * t_c / (a * m_c)

    # Calculate chi
    chi = np.pi / 2 - 2 * np.arctan(t)

    # Calculate lat
    lat = chi + (e ** 2 / 2 + 5 * e ** 4 / 24 + e ** 6 / 12 + 13 * e ** 8 / 360) * np.sin(2 * chi) \
          + (7 * e ** 4 / 48 + 29 * e ** 6 / 240 + 811 * e ** 8 / 11520) * np.sin(4 * chi) \
          + (7 * e ** 6 / 120 + 81 * e ** 8 / 1120) * np.sin(6 * chi) \
          + (4279 * e ** 8 / 161280) * np.sin(8 * chi)

    # Calculate lon
    lon = lambda_0 + np.arctan2(x, -y)

    # Correct the signs and phasing
    lat = -lat
    lon = -lon
    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    # Convert back to degrees
    lat = lat * 180 / np.pi
    lon = lon * 180 / np.pi

    # Make two-column format if user requested no outputs
    if 'nargout' in kwargs and kwargs['nargout'] == 0:
        return np.column_stack((lat, lon))

    return np.round(lat, 4), np.round(lon, 4)


# In[ ]:


def islatlon(lat, lon):
    """
    Determines whether lat, lon is likely to represent geographical
    coordinates.

    Args:
        lat (numpy array): Array of latitudes.
        lon (numpy array): Array of longitudes.

    Returns:
        bool: True if all values in lat are numeric between -90 and 90 inclusive, and all values in lon are numeric between -180 and 360 inclusive.
    """
    if not (isinstance(lat, np.ndarray) and isinstance(lon, np.ndarray)):
        raise ValueError("Input lat and lon must be numpy arrays.")
    
    if not np.all(np.isnan(lat) == False):
        return False
    
    if not np.all(np.isnan(lon) == False):
        return False
    
    if not (np.all(lat >= -90) and np.all(lat <= 90)):
        return False
    
    if not (np.all(lon >= -180) and np.all(lon <= 360)):
        return False
    
    return True


# In[ ]:


def vxvy2uv(lat_or_x, lon_or_y, vx, vy):
    """
    vxvy2uv transforms polar stereographic vector components to
    georeferenced (zonal and meridional) vector components.
    """
    # Input checks
    assert lat_or_x.shape == lon_or_y.shape == vx.shape == vy.shape, "All inputs must be of equal dimensions"
    assert np.issubdtype(lat_or_x.dtype, np.number), "All inputs must be numeric"
    
    # Determine whether inputs are geo coordinates or polar stereographic meters
    if np.all(np.abs(lat_or_x) <= 90) and np.all(np.abs(lon_or_y) <= 180):
        lon = lon_or_y
    else:
        lon, _ = ps2ll(lat_or_x, lon_or_y)  # you need to implement the ps2ll function
        
    # Perform coordinate transformations
    u = vx * np.cos(np.radians(lon)) - vy * np.sin(np.radians(lon))
    v = vy * np.cos(np.radians(lon)) + vx * np.sin(np.radians(lon))
    
    return u, v


# In[ ]:


def quivermc(x, y, u, v, scale, arrow, cmap=None, colorbarOn=False, units=''):
    # Create a quiver plot to visualize the vectors
    quiver_plot = plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=scale, width=arrow)

    # If the user has requested a color map for the arrows
    if cmap is not None:
        magnitude = np.sqrt(u**2 + v**2)
        quiver_plot.set_array(magnitude)
        quiver_plot.set_cmap(cmap)
        if colorbarOn:
            plt.colorbar(quiver_plot, label=units)

    # Show the plot
    plt.show()


# In[ ]:


def thickness2freeboard(T, **kwargs):
    """thickness2freeboard estimates freeboard height above sea level, from ice thickness 
    assuming hyrostatic equilibrium. """
    rhoi = kwargs.get('rhoi', 917)
    rhow = kwargs.get('rhow', 1027)
    rhos = kwargs.get('rhos', 350)
    Ts = kwargs.get('Ts', 0)
    F = (T + Ts * (rhow - rhos) / (rhow - rhoi)) / (rhow / (rhow - rhoi)) # perform the calculation
    return round(F, 2)


# In[ ]:


def freeboard2thickness(F, **kwargs):
    """freeboard2thickness estimates ice thickness from height above sea level,
    assuming hyrostatic equilibrium. """
    rhoi = kwargs.get('rhoi', 917)
    rhow = kwargs.get('rhow', 1027)
    rhos = kwargs.get('rhos', 350)
    Ts = kwargs.get('Ts', 0)
    T = (F * rhow / (rhow - rhoi)) - Ts * (rhow - rhos) / (rhow - rhoi)
    return round(T, 2)


# In[ ]:


def base2freeboard(B, rhoi=917, rhow=1027, rhos=350, Ts=0):
    """
    Estimates freeboard height above sea level, from ice basal elevation,
    assuming hydrostatic equilibrium.

    Parameters:
    B (float): Basal elevation of ice in meters.
    rhoi (float): Ice density in kg/m^3. Default is 917 kg/m^3.
    rhow (float): Water density in kg/m^3. Default is 1027 kg/m^3.
    rhos (float): Snow density in kg/m^3. Default is 350 kg/m^3.
    Ts (float): Snow thickness in meters. Default is 0 m.

    Returns:
    float: Freeboard height above sea level in meters.
    """
    F = (B - Ts * ((rhow - rhos) / (rhow - rhoi))) / (1 - rhow / (rhow - rhoi))

    if F < 0:
        return float("NaN")  # Assume any base elevations above sea level are error or rock
    else:
        return round(F, 2)


# In[ ]:


def contourfps(lat, lon, Z, levels=None, plot_km=False, meridian=0, cmap='viridis'):
    # Convert lat, lon to polar stereographic coordinates
    x, y = ll2ps(lat, lon)

    # Convert to kilometers if user requested:
    if plot_km:
        x = x / 1000
        y = y / 1000

    # Create the contour plot
    fig, ax = plt.subplots()
    if levels is not None:
        cs = ax.contourf(x, y, Z, levels=levels, cmap=cmap)
    else:
        cs = ax.contourf(x, y, Z, cmap=cmap)
        
    plt.colorbar(cs)

    plt.show()


# In[ ]:


def contourps(lat, lon, Z, n=None, v=None, line_spec=None, plot_km=False, meridian=0):
    # Convert lat, lon to polar stereographic coordinates
    x, y = ll2ps(lat, lon)

    # Convert to kilometers if requested:
    if plot_km:
        x = x / 1000
        y = y / 1000

    # Create the contour plot
    fig, ax = plt.subplots()

    if n is not None:
        cs = ax.contour(x, y, Z, n)
    elif v is not None:
        cs = ax.contour(x, y, Z, v)
    else:
        cs = ax.contour(x, y, Z)
    
    if line_spec is not None:
        for line in cs.collections:
            line.set_linestyle(line_spec)

    plt.show()
    return cs


# In[ ]:


def find2drange(X, Y, xi, yi, extraIndices=(0, 0)):
    assert np.issubdtype(X.dtype, np.number), 'X must be numeric.'
    assert X.ndim <= 2, 'This function only works for 1D or 2D X and Y arrays.'
    assert X.shape == Y.shape, 'X and Y must be the same exact size.'
    assert np.issubdtype(xi.dtype, np.number), 'xi must be numeric.'
    assert np.issubdtype(yi.dtype, np.number), 'yi must be numeric.'
    
    extrarows, extracols = extraIndices
    assert extrarows >= 0, 'extrarows must be a positive integer.'
    assert extracols >= 0, 'extracols must be a positive integer.'

    rowsin, colsin = X.shape

    if len(xi) == 0:
        xi = [np.min(X), np.max(X)]
    else:
        xi = [np.min(xi), np.max(xi)]

    if len(yi) == 0:
        yi = [np.min(Y), np.max(Y)]
    else:
        yi = [np.min(yi), np.max(yi)]

    rowi, coli = np.where((X >= xi[0]) & (X <= xi[1]) & (Y >= yi[0]) & (Y <= yi[1]))

    rowrange = np.arange(np.min(rowi) - 1 - extrarows, np.max(rowi) + 2 + extrarows)
    colrange = np.arange(np.min(coli) - 1 - extracols, np.max(coli) + 2 + extracols)

    rowrange = rowrange[(rowrange >= 0) & (rowrange < rowsin)]
    colrange = colrange[(colrange >= 0) & (colrange < colsin)]

    return rowrange, colrange


# In[ ]:


def geoquadps(latlim, lonlim, meridian=0, plotkm=False, **kwargs):
    assert len(latlim) == 2 and len(lonlim) == 2, "Error: latlim and lonlim must each be two-element arrays."
    assert all(-90 <= lat <= 90 for lat in latlim) and all(-180 <= lon <= 180 for lon in lonlim), "Error: latlim and lonlim must be geographic coordinates."

    if np.diff(lonlim) < 0:
        lonlim[1] = lonlim[1] + 360

    lat = np.concatenate((np.linspace(latlim[0], latlim[0], 200), np.linspace(latlim[1], latlim[1], 200), [latlim[0]]))
    lon = np.concatenate((np.linspace(lonlim[0], lonlim[1], 200), np.linspace(lonlim[1], lonlim[0], 200), [lonlim[0]]))

    x, y = ll2ps(lat, lon)

    if plotkm:
        x = x / 1000
        y = y / 1000

    h = plt.plot(x, y, **kwargs)
    plt.gca().set_aspect('equal', adjustable='box')

    return h


def psgrid(CenterLat = None,CenterLon = None,w_km = None,r_km = None, stereographic = False):
        if isinstance(CenterLat,(float,int,np.ndarray)):

             
            if isinstance(CenterLat,(float,int)):
                CenterLat = np.array([CenterLat])
            
            if isinstance(CenterLon,(float,int)):
                CenterLon = np.array([CenterLon])
            
            if islatlon(CenterLat,CenterLon):
                [centerx,centery] = ll2ps(CenterLat,CenterLon)
            else:
                centerx = CenterLat
                centery = CenterLon
                
            width_km= w_km
            resolution_km =r_km
           
        else:
            [centerx,centery] = scarloc(CenterLat,'xy')
            width_km= w_km
            resolution_km =r_km
            
        if isinstance (width_km,(float,int)):
            width_km = [width_km]
            
        if isinstance (resolution_km,(float,int)):
            resolution_km = [resolution_km]
            
        if len(width_km) == 1:
            widthx = width_km[0]*1000 # The *1000 bit converts from km to meters. 
            widthy = width_km[0]*1000
            
        elif len(width_km) == 2:
            widthx = width_km[0]*1000 
            widthy = width_km[1]*1000
        else:
            raise ValueError("I must have misinterpreted something. As I understand it, you have requested a grid width with more than two elements. Check inputs and try again.")
           
        
        if len(resolution_km) == 1:
            resx = resolution_km[0]*1000
            resy = resolution_km[0]*1000
        
        elif len(resolution_km) == 2:
            resx = resolution_km[0]*1000
            rexy = resolution_km[1]*1000
        else: 
            raise ValueError("I must have misinterpreted something. As I understand it, you have requested a grid resolution with more than two elements. Check inputs and try again.") 
            

        # Verify that resolution is not greater than width
        assert widthx > resx, "It looks like there's an input error because the grid width should be bigger than the grid resolution. Check inputs and try again."
        assert widthy > resy, "It looks like there's an input error because the grid width should be bigger than the grid resolution. Check inputs and try again."
        assert resx > 0, "Grid resolution must be greater than zero."
        assert resy > 0, "Grid resolution must be greater than zero."
        assert widthx > 0, "Grid width must be greater than zero."
        assert widthy > 0, "Grid width must be greater than zero."

        # Should outputs be polar stereographic?

        outputps = (stereographic == 'xy')

        # Build grid
        x = np.arange(centerx - widthx/2, centerx + widthx/2 + resx, resx)
        y = np.arange(centery - widthy/2, centery + widthy/2 + resy, resy)
        X, Y = np.meshgrid(x, y)
    

        # Convert coordinates if necessary
        if outputps:
            out1 = X
            out2 = Y
            
        else:
            out1, out2 = ps2ll(X, Y)

        return out1, out2
    
#test functions ps grid:
#psgrid(-75,-107,600,5)    
#psgrid('amery ice shelf',w_km = [900,250],r_km = 10,stereographic = 'xy')            
 
