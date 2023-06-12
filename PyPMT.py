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
    phi_c = -71  # degrees
    a = 6378137.0  # meters
    e = 0.08181919
    lambda_0 = 0  # degrees

    # Parse optional keyword arguments
    for key, value in kwargs.items():
        if key == 'TrueLat':
            phi_c = value
        elif key == 'EarthRadius':
            a = value
        elif key == 'Eccentricity':
            e = value
        elif key == 'meridian':
            lambda_0 = value

    # Convert to radians and switch signs
    phi_c = -phi_c * math.pi / 180
    lambda_0 = -lambda_0 * math.pi / 180
    x = -x
    y = -y

    # Calculate constants
    t_c = math.tan(math.pi / 4 - phi_c / 2) / ((1 - e * math.sin(phi_c)) / (1 + e * math.sin(phi_c))) ** (e / 2)
    m_c = math.cos(phi_c) / math.sqrt(1 - e ** 2 * (math.sin(phi_c)) ** 2)

    # Calculate rho and t
    rho = np.sqrt(x ** 2 + y ** 2)
    t = rho * t_c / (a * m_c)

    # Calculate chi
    chi = math.pi / 2 - 2 * math.atan(t)

    # Calculate lat
    lat = chi + (e ** 2 / 2 + 5 * e ** 4 / 24 + e ** 6 / 12 + 13 * e ** 8 / 360) * math.sin(2 * chi) \
          + (7 * e ** 4 / 48 + 29 * e ** 6 / 240 + 811 * e ** 8 / 11520) * math.sin(4 * chi) \
          + (7 * e ** 6 / 120 + 81 * e ** 8 / 1120) * math.sin(6 * chi) \
          + (4279 * e ** 8 / 161280) * math.sin(8 * chi)

    # Calculate lon
    lon = lambda_0 + np.arctan2(x, -y)

    # Correct the signs and phasing
    lat = -lat
    lon = -lon
    lon = (lon + math.pi) % (2 * math.pi) - math.pi

    # Convert back to degrees
    lat = lat * 180 / math.pi
    lon = lon * 180 / math.pi

    # Make two-column format if user requested no outputs
    if 'nargout' in kwargs and kwargs['nargout'] == 0:
        return np.column_stack((lat, lon))

    return round(lat, 4), round(lon, 4)


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

