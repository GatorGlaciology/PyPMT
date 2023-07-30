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
from geopy.distance import geodesic
from pyproj import CRS, Transformer
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d
from shapely import MultiPoint
from shapely.geometry import LineString
from shapely.ops import shared_paths
import mpl_toolkits
from mpl_toolkits.basemap import Basemap

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


def ll2ps(lat, lon, **kwargs):
    # Set default values
    phi_c = kwargs.get('TrueLat', -71)
    a = kwargs.get('EarthRadius', 6378137.0)
    e = kwargs.get('Eccentricity', 0.08181919)
    lambda_0 = kwargs.get('meridian', 0)

    # Convert degrees to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lambda_0_rad = np.deg2rad(lambda_0)
    phi_c_rad = np.deg2rad(phi_c)

    # Calculate m and t values
    m_c = np.cos(phi_c_rad) / np.sqrt(1 - e ** 2 * (np.sin(phi_c_rad) ** 2))
    t_c = np.tan(np.pi / 4 - phi_c_rad / 2) / ((1 - e * np.sin(phi_c_rad)) / (1 + e * np.sin(phi_c_rad))) ** (e / 2)
    t = np.tan(np.pi / 4 - lat_rad / 2) / ((1 - e * np.sin(lat_rad)) / (1 + e * np.sin(lat_rad))) ** (e / 2)

    # Calculate rho value
    rho = a * m_c * t_c / t

    # Calculate x and y
    x = rho * np.sin(lon_rad - lambda_0_rad)
    y = rho * np.cos(lon_rad - lambda_0_rad)

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
            if not np.isscalar(phi_c):
                raise ValueError('True lat must be a scalar.')
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


def geoquadps(m, latlim, lonlim, meridian=0, plotkm=False, **kwargs):
    assert len(latlim) == 2 and len(lonlim) == 2, "Error: latlim and lonlim must each be two-element arrays."
    assert all(-90 <= lat <= 90 for lat in latlim) and all(-180 <= lon <= 180 for lon in lonlim), "Error: latlim and lonlim must be geographic coordinates."

    if np.diff(lonlim) < 0:
        lonlim[1] = lonlim[1] + 360

    lat = np.concatenate((np.linspace(latlim[0], latlim[0], 200), np.linspace(latlim[1], latlim[1], 200), [latlim[0]]))
    lon = np.concatenate((np.linspace(lonlim[0], lonlim[1], 200), np.linspace(lonlim[1], lonlim[0], 200), [lonlim[0]]))

    x, y = m(lat, lon)

    if plotkm:
        x = x / 1000
        y = y / 1000

    m = plt.plot(x, y, **kwargs)

    return m


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

    if isinstance(width_km,(float,int)):
        width_km = [width_km]

    if isinstance(resolution_km,(float,int)):
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


def uv2vxvy(lat_or_x, lon_or_y, u, v):
    # Convert inputs to numpy arrays if they are not already
    lat_or_x = np.array(lat_or_x) if not isinstance(lat_or_x, np.ndarray) else lat_or_x
    lon_or_y = np.array(lon_or_y) if not isinstance(lon_or_y, np.ndarray) else lon_or_y
    u = np.array(u) if not isinstance(u, np.ndarray) else u
    v = np.array(v) if not isinstance(v, np.ndarray) else v

    # Input checks
    assert isinstance(lat_or_x, (int, float, np.ndarray)), 'All inputs for uv2vxvy must be numeric.'
    assert isinstance(lon_or_y, (int, float, np.ndarray)), 'All inputs for uv2vxvy must be numeric.'
    assert isinstance(u, (int, float, np.ndarray)), 'All inputs for uv2vxvy must be numeric.'
    assert isinstance(v, (int, float, np.ndarray)), 'All inputs for uv2vxvy must be numeric.'
    assert np.shape(lat_or_x) == np.shape(lon_or_y) == np.shape(u) == np.shape(
        v), 'All inputs to uv2vxvy must be of equal dimensions.'

    # Parse inputs
    if islatlon(lat_or_x, lon_or_y):
        lon = lon_or_y  # lat is really just a placeholder to make the function a little more intuitive to use. It is not necessary for calculation.
    else:
        _, lon = ps2ll(lat_or_x, lon_or_y)

    # Convert lon to radians
    lon_rad = np.deg2rad(lon)

    # Perform calculation
    vx = u * np.cos(lon_rad) + v * np.sin(lon_rad)
    vy = -u * np.sin(lon_rad) + v * np.cos(lon_rad)

    return vx, vy


def pathdist(lat, lon, units='m', track='gc', refpoint=None):
    assert len(lat) == len(lon), 'Length of lat and lon must match.'
    assert len(lat) > 1, 'lat and lon must have more than one point.'

    # Check if reference point is defined:
    if refpoint is not None:
        assert len(
            refpoint) == 2, 'Coordinates of reference point can be only a single point given by a latitude/longitude pair in the form [reflat reflon].'

    # Convert units to geopy format
    if units in ['m', 'meter(s)', 'metre(s)']:
        units = 'meters'
    elif units in ['km', 'kilometer(s)', 'kilometre(s)']:
        units = 'kilometers'
    elif units in ['nm', 'naut mi', 'nautical mile(s)']:
        units = 'nautical'
    elif units in ['ft', 'international ft', 'foot', 'international foot', 'feet', 'international feet']:
        units = 'feet'
    elif units in ['in', 'inch', 'inches']:
        units = 'inches'
    elif units in ['yd', 'yds', 'yard(s)']:
        units = 'yards'
    elif units in ['mi', 'mile(s)', 'international mile(s)']:
        units = 'miles'

    # Initialize path distance
    path_distance = [0]
    ref_distance = 0

    # Calculate distance between each pair of points
    for i in range(1, len(lat)):
        start_point = (lat[i - 1], lon[i - 1])
        end_point = (lat[i], lon[i])

        # Calculate the geodesic distance between the points
        distance = geodesic(start_point, end_point).meters

        # If this is the reference point, update the reference distance
        if refpoint is not None and (lat[i - 1], lon[i - 1]) == tuple(refpoint):
            ref_distance = sum(path_distance)

        # If units are not meters, convert distance to specified units
        if units != 'meters':
            if units == 'kilometers':
                distance = distance / 1000
            elif units == 'miles':
                distance = distance / 1609.34  # 1 mile is approximately 1609.34 meters
            elif units == 'feet':
                distance = distance / 0.3048  # 1 foot is approximately 0.3048 meters

        # Add the distance to the previous cumulative distance
        path_distance.append(path_distance[-1] + distance - ref_distance)
    return path_distance


def inpsquad(lat, lon, latlim, lonlim, inclusive=False):
    assert np.array(lat.shape) == np.array(lon.shape), 'Inputs lat and lon must be the same size.'
    assert np.array(latlim.shape) == np.array(
        lonlim.shape), 'Inputs latlim_or_xlim and lonlim_or_ylim must be the same size.'
    assert len(latlim) > 1, 'latlim or xlim must have more than one point.'

    min_lat, max_lat = min(latlim), max(latlim)
    min_lon, max_lon = min(lonlim), max(lonlim)

    IN = np.logical_and(lat >= min_lat, lat <= max_lat)
    IN = np.logical_and(IN, lon >= min_lon)
    IN = np.logical_and(IN, lon <= max_lon)

    return IN


def psdistortion(lat, true_lat=-71):
    assert np.all(np.abs(lat) <= 90), 'Error: inputs must be latitudes.'
    assert np.isscalar(true_lat), 'Error: true_lat must be a scalar.'
    assert np.abs(true_lat) <= 90, 'Error: true_lat must be in the range -90 to 90.'

    lat = np.radians(lat)  # convert from degrees to radians
    true_lat = np.radians(true_lat)  # same for true_lat

    # calculate map scale factor
    m = (1 + np.sin(np.abs(true_lat))) / (1 + np.sin(np.abs(lat)))
    return m


def pathdistps(lat_or_x, lon_or_y, *args):
    # Initialize variables
    lat_or_x = np.array(lat_or_x)
    lon_or_y = np.array(lon_or_y)
    kmout = False
    ref = False
    refcoord = None

    # Parse optional arguments
    for arg in args:
        if arg == 'km':
            kmout = True
        elif isinstance(arg, list) and len(arg) == 2:
            ref = True
            refcoord = arg

    # Convert geo coordinates to polar stereographic if necessary
    if islatlon(lat_or_x, lon_or_y):
        lat = lat_or_x
        [x, y] = ll2ps(lat_or_x, lon_or_y)
    else:
        x = lat_or_x
        y = lon_or_y
        lat, _ = ps2ll(x, y) #don't need lon

    # Perform mathematics:
    m = psdistortion(lat[1:])  # Assuming psdistortion is defined or imported

    # Cumulative sum of distances:
    d = np.zeros_like(x)
    d[1:] = np.cumsum(np.hypot(np.diff(x)/m, np.diff(y)/m))

    # Reference to a location
    if ref and refcoord is not None:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031")
        ref_x, ref_y = transformer.transform(refcoord[0], refcoord[1])
        dist_to_refpoint = np.hypot(x - ref_x, y - ref_y)
        min_dist_index = np.argmin(dist_to_refpoint)
        d = d - d[min_dist_index]

    # Convert to kilometers if user wants it that way
    if kmout:
        d = d / 1000

    return d


def pspath(lat_or_x, lon_or_y, spacing, method='linear'):
    assert isinstance(lat_or_x, np.ndarray) and lat_or_x.ndim == 1, 'Input error: input coordinates must be vectors of matching dimensions.'
    assert lat_or_x.shape == lon_or_y.shape, 'Input error: dimensions of input coordinates must match.'
    assert np.isscalar(spacing), 'Input error: spacing must be a scalar.'

    geoin = islatlon(lat_or_x, lon_or_y)
    if geoin:
        x, y = ll2ps(lat_or_x, lon_or_y)
    else:
        x = lat_or_x
        y = lon_or_y

    d = pathdistps(x, y)

    # Create interpolation function based on method
    func_x = interp1d(d, x, kind=method, fill_value="extrapolate")
    func_y = interp1d(d, y, kind=method, fill_value="extrapolate")

    # Generate equally spaced array from 0 to max(d)
    d_new = np.arange(0, d[-1], spacing)

    xi = func_x(d_new)
    yi = func_y(d_new)

    # Convert to geo coordinates if inputs were geo coordinates
    if geoin:
        out1, out2 = ps2ll(xi, yi)
    else:
        out1 = xi
        out2 = yi

    return out1, out2


def pathcrossingps71(lat1, lon1, lat2, lon2, clip_option=None):
    assert isinstance(lat1, list) and all(isinstance(i, float) for i in lat1), 'Input lat1 must be a list of floats.'
    assert len(lat1) == len(lon1), 'Input lat1 and lon1 must be the same size.'
    assert isinstance(lat2, list) and all(isinstance(i, float) for i in lat2), 'Input lat2 must be a list of floats.'
    assert len(lat2) == len(lon2), 'Input lat2 and lon2 must be the same size.'

    clip_data = True
    if clip_option is not None:
        if clip_option.lower().startswith('no') or clip_option.lower() == 'off':
            clip_data = False

    # Transform to polar stereo coordinates with standard parallel at 71 S
    # Here we assume that ll2ps and ps2ll are already defined functions
    x1, y1 = ll2ps(lat1, lon1)
    x2, y2 = ll2ps(lat2, lon2)

    # Delete faraway points before performing InterX function for large data sets
    # This part of code is omitted for brevity and because it is an optimization
    if clip_data:
        if len(x1) * len(x2) > 1e6:
            for _ in range(2):
                stdx1 = np.std(np.diff(x1))
                stdy1 = np.std(np.diff(y1))
                stdx2 = np.std(np.diff(x2))
                stdy2 = np.std(np.diff(y2))
                x1, y1 = clip_outliers(x1, y1, x2, stdy1, 'x')
                x2, y2 = clip_outliers(x2, y2, x1, stdy2, 'x')
                x1, y1 = clip_outliers(x1, y1, y2, stdy1, 'y')
                x2, y2 = clip_outliers(x2, y2, y1, stdy2, 'y')

    # Find intersection x,y point(s)
    # Here we assume that InterX is already defined function
    P = InterX([x1, y1], [x2, y2])

    # If InterX returns None, try using shared_paths
    if P is None:
        line1 = LineString(np.column_stack([x1, y1]))
        line2 = LineString(np.column_stack([x2, y2]))
        shared = shared_paths(line1, line2)
        if not shared.is_empty:
            # Access the shared paths via the 'geoms' attribute
            forward, reverse = shared.geoms
            if not forward.is_empty:
                # Extract the coordinates of each sub-geometry in 'forward'
                coords = []
                for geom in forward.geoms:
                    coords.extend(geom.coords)
                x, y = np.array(coords).T
                P = [x, y]

    # If P is still None after trying shared_paths, return None
    if P is None:
        return None

    # Transform back to lat/lon space
    lati, loni = ps2ll(np.array(P[0]), np.array(P[1]))

    return lati, loni


def clip_outliers(x, y, x_compare, std, axis):
    if axis == 'x':
        y = y[np.abs(x - np.mean(x_compare)) < std]
        x = x[np.abs(x - np.mean(x_compare)) < std]
    elif axis == 'y':
        x = x[np.abs(y - np.mean(x_compare)) < std]
        y = y[np.abs(y - np.mean(x_compare)) < std]
    return x, y


def InterX(L1, L2):
    line1 = LineString(np.column_stack(L1))
    line2 = LineString(np.column_stack(L2))
    intersection = line1.intersection(line2)
    if intersection.is_empty:
        return None
    elif intersection.geom_type == 'Point':
        x, y = intersection.xy
        return [list(x), list(y)]

    elif intersection.geom_type == 'MultiPoint':
        x, y = MultiPoint(intersection).xy
        return [list(x), list(y)]

    elif intersection.geom_type == 'GeometryCollection':
        intersections = [np.column_stack((point.xy)) for point in intersection if point.geom_type == 'Point']
        return intersections


def antbounds():
    fig, ax = plt.subplots(figsize=(10,10))
    m = Basemap(projection='spstere',boundinglat=-60,lon_0=180,resolution='c')
    m.drawcoastlines()
    return m, ax