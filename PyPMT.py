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




