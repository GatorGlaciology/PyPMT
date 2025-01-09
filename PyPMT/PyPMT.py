#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from random import choice, seed
import csv
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from pyproj import Transformer
from scipy.interpolate import interp1d
from shapely import MultiPoint
from shapely.geometry import LineString
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import re
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
from pathlib import Path
import geopandas as gpd
from importlib.resources import files

# In[ ]:


def load_csv_data(file_name):
    """
    Load in a csv file (used in scarloc)

    Parameters
    ----------
        file_name : string
            path to the file

    Returns
    -------
        numpy.ndarrays for latitude, longitude, and location names
    """

    # Resolve the path relative to the current file's location
    data_dir = Path(__file__).resolve().parent / 'data'
    file_path = data_dir / file_name

    lat = []
    lon = []
    names = []

    with open(file_path, 'r', encoding='utf-8') as csvfile:
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


def str_lookup(feature_name, names):
    """
    Search for features by name (used in scarloc)

    Parameters
    ----------
        feature_name : string
            feature we will be searching for
        names : np.ndarray
            names of features (loaded in with load_csv_data)

    Returns
    -------
        x : int
            integer of the first index of names that matches feature_name
        nearby_names : list
            items from names that match feature_name
    """

    feature_name = feature_name.lower()  # Convert feature_name to lowercase for case-insensitive matching

    indices = [i for i, name in enumerate(names) if feature_name in name.lower()]

    if len(indices) > 0:
        x = indices[0]  # Choose the first matching index
        nearby_names = [names[i] for i in indices]  # Get all matching names
    else:
        x = None
        nearby_names = []

    return x, nearby_names


def scar_loc(feature_name, *varargin):
    """
    Find lat/lon coordinates of Antarctic features

    Parameters
    ----------
        feature_name : string
            name of the desired feature

    Returns
    -------
        varargout : numpy.ndarray or list
            latitude and longitude coordinates
    """

    # Check if feature_name is a list or a single string
    if isinstance(feature_name, str):
        feature_name = [feature_name]

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

    feature_lat = np.full(len(feature_name), np.nan)
    feature_lon = np.full(len(feature_name), np.nan)

    # Look for each feature name
    for k in range(len(feature_lat)):
        x, nearby_names = str_lookup(feature_name[k], names)
        if x is None and OfferHelp:
            fmsg = [
                f'"{feature_name[k]}" not found.',
                f'Are you sure that "{feature_name[k]}" exists in Antarctica?',
                'Did a cat walk across your keyboard?',
                'This is the real reason one shouldn''t text and drive. Check your spelling and try again.',
                'Now you''re just making things up.',
                f'SCAR has identified more than 25,000 features in Antarctica, but "{feature_name[k]}" is not one of them.',
                f'Can''t find "{feature_name[k]}".',
                f'"{feature_name[k]}" may exist somewhere in the world, but you won''t find it in Antarctica.',
                f'It is possible that Robert F. Scott named something in Antarctica "{feature_name[k]}", but if he did there are no records of it.',
                f'You must be thinking of {feature_name[k]}, Kansas, because {feature_name[k]}, Antarctica does not exist.',
                f'Sure, they used to just call it {feature_name[k]}, but not anymore, what with political correctness and all.',
                f'"{feature_name[k]}" is an interesting combination of letters, but I don''t think it''s any place in Antarctica.',
                f'The great Wayne Cochran once sang, "Where oh where can my {feature_name[k]} be?" Because it''s not in Antarctica.',
                f'I''m pretty sure it is in violation of the Antarctic Treaty to refer to any place as "{feature_name[k]}".',
                f'"{feature_name[k]}" does not match any entries in the SCAR database.',
                f'Science is all about formality, so the bigwigs will surely look down their noses at such colloquial jargon as "{feature_name[k]}".',
                f'My doctor said I need to get my {feature_name[k]} removed.',
                'Frostbitten Antarctic researcher mistypes again.',
                'This may be an issue of American English versus British English.',
                f'Antarctica''s a strange place, but it''s not science fiction. Verify that "{feature_name[k]}" actually exists.',
                f'What''s in a name? I''ll tell you what''s in a name: That which you call "{feature_name[k]}" by any other name may actually exist in Antarctica.',
                f'Did John Carpenter tell you''ll find "{feature_name[k]}" in Antarctica?',
                f'You know, some folks say glaciology is a shrinking field, but I say things are just heating up. In other news, "{feature_name[k]}" does not exist.',
                f'You''re a glaciologist? Isn''t that a slow-moving field? Also, I have to tell you, I can''t seem to find any record of "{feature_name[k]}".',
                f'Amazing glaciology, how sweet the sound... "{feature_name[k]}" once was lost, and still has not been found.'
            ]

            np.random.shuffle(fmsg)
            print(fmsg[0])
            if nearby_names:
                print('Here are the best matches I can find:')
                print(nearby_names)
            else:
                print('Try typing "load scarnames" to explore the available list of features.')
            return

        if x is not None:
            feature_lat[k] = lat[x]
            feature_lon[k] = lon[x]
    
    # Convert to polar stereographic coordinates
    if pscoords:
        feature_lat, feature_lon = ll2ps(feature_lat, feature_lon)

    # Convert to polar stereographic kilometers
    if kmout:
        feature_lon = feature_lon / 1000
        feature_lat = feature_lat / 1000

    # Returning only latitude or only x would not make any sense,
    # so if no outputs are requested, or if only one output is requested,
    # return as a lat column and lon column or [x y]
    if len(feature_name) == 1:
        varargout = np.column_stack((feature_lat, feature_lon))
        return varargout[0]
    else:
        varargout = [feature_lat, feature_lon]
        return varargout

# In[ ]:


def handle_missing_feature(feature_name, nearby_names):
    """
    Prints messages to notify users that feature_name is not in the names array.

    Parameters
    ----------
        feature_name : string
            name of the desired feature
        nearby_names : list
            items from names that match feature_name
    """
    fmsg = [
        f'"{feature_name}" not found.',
        f'Are you sure that "{feature_name}" exists in Antarctica?',
        'Did a cat walk across your keyboard?',
        'This is the real reason one shouldn\'t text and drive. Check your spelling and try again.',
        'Now you\'re just making things up.',
        f'SCAR has identified more than 25,000 features in Antarctica, but "{feature_name}" is not one of them.',
        f'Can\'t find "{feature_name}".',
        f'"{feature_name}" may exist somewhere in the world, but you won\'t find it in Antarctica.',
        f'It is possible that Robert F. Scott named something in Antarctica "{feature_name}", but if he did there are no records of it.',
        f'You must be thinking of {feature_name}, Kansas, because {feature_name}, Antarctica does not exist.',
        f'Sure, they used to just call it {feature_name}, but not anymore, what with political correctness and all.',
        f'"{feature_name}" is an interesting combination of letters, but I don\'t think it\'s any place in Antarctica.',
        f'The great Wayne Cochran once sang, "Where oh where can my {feature_name} be?" Because it\'s not in Antarctica.',
        f'I\'m pretty sure it is in violation of the Antarctic Treaty to refer to any place as "{feature_name}".',
        f'"{feature_name}" does not match any entries in the SCAR database.',
        f'Science is all about formality, so the bigwigs will surely look down their noses at such colloquial jargon as "{feature_name}".',
        f'My doctor said I need to get my {feature_name} removed.',
        'Frostbitten Antarctic researcher mistypes again.',
        'This may be an issue of American English versus British English.',
        f'Antarctica\'s a strange place, but it\'s not science fiction. Verify that "{feature_name}" actually exists.',
        f'What\'s in a name? I\'ll tell you what\'s in a name: That which you call "{feature_name}" by any other name may actually exist in Antarctica.',
        f'Did John Carpenter tell you\'ll find "{feature_name}" in Antarctica?',
        f'You know, some folks say glaciology is a shrinking field, but I say things are just heating up. In other news, "{feature_name}" does not exist.',
        f'You\'re a glaciologist? Isn\'t that a slow-moving field? Also, I have to tell you, I can\'t seem to find any record of "{feature_name}".',
        f'Amazing glaciology, how sweet the sound... "{feature_name}" once was lost, and still has not been found.'
    ]

    rngstart = seed()  # get initial rng setting before changing it temporarily.
    random_msg = choice(fmsg)
    print(random_msg)
    seed(rngstart)  # returns to original rng settings.

    if nearby_names:
        print('Here are the best matches I can find:')
        print(nearby_names)
    else:
        print('Try typing "load scarnames" to explore the available list of features.')

    return np.nan, np.nan


# In[ ]:


def ll2ps(lat, lon, **kwargs):
    """
    Converts latitude/longitude coordinates to map coordinates for a polar stereographic system.

    Parameters
    ----------
        lat : float, int, list of floats/int, or numpy ndarray of floats/int
            latitude coordinate(s)
        lon : float, int, list of floats/int, or numpy ndarray of floats/int
            longitude coordinate(s)

    Returns
    -------
        x : float or list of floats
            calculated x coordinate(s)
        y : float or list of floats
            calculated y coordinate(S)
    """
    # Set default values
    phi_c = -71
    a = 6378137.0
    e = 0.08181919
    lambda_0 = 0

    if not is_lat_lon(lat, lon):
        raise ValueError('Please verify your input latitude and longitude coordinates.')

    # Parse optional keyword arguments
    for key, value in kwargs.items():
        if key.lower() == 'true_lat':
            phi_c = value
            if not np.isscalar(phi_c):
                raise ValueError('True lat must be a scalar.')
            """if phi_c > 0:
                print("I'm assuming you forgot the negative sign for the true latitude, \
                      and I am converting your northern hemisphere value to southern hemisphere.")
                phi_c = -phi_c"""
        elif key.lower() == 'earth_radius':
            a = value
            if not isinstance(a, (int, float)):
                raise ValueError('Earth radius must be a scalar.')
            if a < 7e+3:
                raise ValueError('Earth radius should be something like 6378137 in kilometers.')
        elif key.lower() == 'eccentricity':
            e = value
            if not isinstance(e, (int, float)):
                raise ValueError('Earth eccentricity must be a scalar.')
            if e < 0 or e >= 1:
                raise ValueError('Earth eccentricity does not seem like a reasonable value.')
        elif key.lower() == 'meridian':
            lambda_0 = value
            if not isinstance(lambda_0, (int, float)):
                raise ValueError('Meridian must be a scalar.')
            if lambda_0 < -180 or lambda_0 > 360:
                raise ValueError('Meridian does not seem like a logical value.')
        else:
            print("At least one of your input arguments is invalid. Please try again.")
            return 0

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

    return np.round(x, 4), np.round(y, 4)


# In[ ]:


def ps2ll(x, y, **kwargs):
    """
    Converts polar stereographic coordinates to latitude/longitude coordinates.

    Parameters
    ----------
        x : float
            calculated x coordinate(s)
        y : float
            calculated y coordinate(S)

    Returns
    -------
        lat : float, int, list of floats/int, or numpy ndarray of floats/int
            latitude coordinate(s)
        lon : float, or numpy ndarray of floats
            longitude coordinate(s)
    """
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
            """if phi_c > 0:
                print("I'm assuming you forgot the negative sign for the true latitude, \
                      and I am converting your northern hemisphere value to southern hemisphere.")
                phi_c = -phi_c"""
        elif key.lower() == 'earth_radius':
            a = value
            if not isinstance(a, (int, float)):
                raise ValueError('Earth radius must be a scalar.')
            if a < 7e+3:
                raise ValueError('Earth radius should be something like 6378137 in kilometers.')
        elif key.lower() == 'eccentricity':
            e = value
            if not isinstance(e, (int, float)):
                raise ValueError('Earth eccentricity must be a scalar.')
            if e <= 0 or e > 1:
                raise ValueError('Earth eccentricity does not seem like a reasonable value.')
        elif key.lower() == 'meridian':
            lambda_0 = value
            if not isinstance(lambda_0, (int, float)):
                raise ValueError('Meridian must be a scalar.')
            if lambda_0 < -180 or lambda_0 > 360:
                raise ValueError('Meridian does not seem like a logical value.')
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


def is_lat_lon(lat, lon):
    """
    Determines whether lat, lon is likely to represent geographical coordinates.

    Parameters
    ----------
        lat : numpy ndarray
            array of latitude coordinates
        lon : numpy ndarray
            array of longitude coordinates

    Returns
    -------
        bool: True if all values in lat are numeric between -90 and 90 inclusive,
         and all values in lon are numeric between -180 and 360 inclusive. False otherwise.
    """

    if not (isinstance(lat, np.ndarray) and isinstance(lon, np.ndarray)):
        lat = np.ndarray(lat)
        lon = np.ndarray(lon)
    
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
    Transforms polar stereographic coordinates to georeferenced (zonal and meridional) coordinates.

    Parameters
    ----------
        lat_or_x : numpy ndarray
            array of latitude or x (polar stereographic) coordinates
        lon_or_y : numpy ndarray
            array of longitude or y (polar stereographic) coordinates
        vx : numpy ndarray
            magnitudes to multiply lat_or_x values by
        vy : numpy ndarray
            magnitudes to multiply lon_or_y values by

    Returns
    -------
        u : numpy ndarray
            the zonal component, representing the eastward component of the vector field
        v : numpy ndarray
            the meridional component, representing the northward component of the vector field
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


def quiver_mc(lat, lon, u, v, **kwargs):
    """
    Plots vectors of zonal and meridional components.

    Parameters
    ----------
        lat : numpy ndarray
            array of latitude coordinates
        lon : numpy ndarray
            array of longitude coordinates
        u : numpy ndarray
            the zonal component, representing the eastward component of the vector field
        v : numpy ndarray
            the meridional component, representing the northward component of the vector field

    Returns
    -------
        Displays a matplotlib plot.
    """

    # Calculate the magnitude of the vectors
    magnitude = np.sqrt(u ** 2 + v ** 2)

    plt.figure(figsize=(20, 20))
    plt.quiver(lon, lat, u, v, magnitude, **kwargs)
    plt.show()


def quiver_ps(lat, lon, u, v, **kwargs):
    """
    Plots vectors of zonal and meridional components.

    Parameters
    ----------
        lat : numpy ndarray
            array of latitude coordinates
        lon : numpy ndarray
            array of longitude coordinates
        u : numpy ndarray
            the zonal component, representing the eastward component of the vector field
        v : numpy ndarray
            the meridional component, representing the northward component of the vector field

    Returns
    -------
        Displays a matplotlib plot.
    """

    # Create a figure and axes
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.SouthPolarStereo()}, figsize=(10, 10))

    # Handle NaN values
    u = np.nan_to_num(u)
    v = np.nan_to_num(v)

    # Calculate the magnitude of the vectors
    magnitude = np.sqrt(u ** 2 + v ** 2)

    # Create a polar stereographic CRS
    crs = ccrs.SouthPolarStereo()

    # Create a transform object
    transform = ccrs.PlateCarree()._as_mpl_transform(ax)

    # Add the quiver plot to the axes
    ax.quiver(lon, lat, u, v, magnitude, transform=ccrs.PlateCarree(), **kwargs)

    # Display the plot.
    plt.show()

# In[ ]:


def thickness_2_freeboard(t, **kwargs):
    """
    Estimates freeboard height above sea level, from ice thickness, assuming hydrostatic equilibrium.

    Parameters
    ----------
        t : int or float
            ice thickness

    Returns
    -------
        f : float
            freeboard height above sea level
    """

    rhoi = kwargs.get('rhoi', 917)
    rhow = kwargs.get('rhow', 1027)
    rhos = kwargs.get('rhos', 350)
    ts = kwargs.get('ts', 0)
    f = (t + ts * (rhow - rhos) / (rhow - rhoi)) / (rhow / (rhow - rhoi))  # perform the calculation
    return round(f, 2)


# In[ ]:


def freeboard_2_thickness(f, **kwargs):
    """
    Estimates ice thickness from freeboard height above sea level, assuming hydrostatic equilibrium.

    Parameters
    ----------
        f : int or float
            freeboard height above sea level

    Returns
    -------
        t : float
            ice thickness
    """
    rhoi = kwargs.get('rhoi', 917)
    rhow = kwargs.get('rhow', 1027)
    rhos = kwargs.get('rhos', 350)
    ts = kwargs.get('ts', 0)
    t = (f * rhow / (rhow - rhoi)) - ts * (rhow - rhos) / (rhow - rhoi)
    return round(t, 2)


# In[ ]:


def base_2_freeboard(B, rhoi=917, rhow=1027, rhos=350, ts=0):
    """
    Estimates freeboard height above sea level, from ice basal elevation,
    assuming hydrostatic equilibrium.

    Parameters
    ----------
        B : float
            Basal elevation of ice (in meters).
        rhoi : float
            Ice density in kg/m^3. Default is 917 kg/m^3.
        rhow : float
            Water density in kg/m^3. Default is 1027 kg/m^3.
        rhos : float
            Snow density in kg/m^3. Default is 350 kg/m^3.
        ts : float
            Snow thickness in meters. Default is 0 m.


    Returns
    -------
        f : float
            Freeboard height above sea level (in meters).
    """
    f = (B - ts * ((rhow - rhos) / (rhow - rhoi))) / (1 - rhow / (rhow - rhoi))

    if f < 0:
        return float("NaN")  # Assume any base elevations above sea level are error or rock
    else:
        return round(f, 2)


# In[ ]:


def contour_ps(lat, lon, z, n=None, v=None, line_spec=None, plot_km=False, ax=None, fill=False):
    """
    Contour function that plots georeferenced data in polar stereographic coordinates.

    Parameters
    ----------
        lat : numpy ndarray
            array of latitude coordinates
        lon : numpy ndarray
            array of longitude coordinates
        z : numpy ndarray
            array of z coordinates
        n : float or int or array-like, optional
            The number of contour levels or an array of contour values. If not specified,
            a default set of contour levels will be used.
        v : float or int or array-like, optional
            Alternatively, you can specify specific contour values using the `v` parameter.
        line_spec : str, optional
            Line style for contour lines (e.g., 'dashed', 'solid', 'dotted'). If not specified,
            default line styles will be used.
        plot_km : bool, optional
            If True, convert latitude and longitude coordinates to kilometers.
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Axes on which to plot the contour. If not specified, a new figure and axes will be created.
        fill : bool, optional
            If True, create filled contours. If False, create contour lines. Default is False.


    Returns
    -------
        cs : matplotlib.contour.QuadContourSet
            A QuadContourSet instance representing the filled or line contours.
    """
    # Convert lat, lon to polar stereographic coordinates
    x, y = lat, lon
    # print(x)

    # Convert to kilometers if requested:
    if plot_km:
        x = x / 1000
        y = y / 1000

    if ax is None:
        # Create the contour plot
        fig, ax = plt.subplots()

    if not fill:
        if n is not None:
            cs = ax.contour(x, y, z, n)
        elif v is not None:
            cs = ax.contour(x, y, z, v)
        else:
            cs = ax.contour(x, y, z)
    else:
        if n is not None:
            cs = ax.contourf(x, y, z, n)
        elif v is not None:
            cs = ax.contourf(x, y, z, v)
        else:
            cs = ax.contourf(x, y, z)

    if line_spec is not None:
        for line in cs.collections:
            line.set_linestyle(line_spec)

    if ax is None:
        plt.show()

    return cs


# In[ ]:


def find_2d_range(x, y, xi, yi, extra_indices=(0, 0)):
    """
    Returns matrix indices which encompass a range of xi, yi values.

    Parameters
    ----------
        x : numpy ndarray
            array of x coordinates
        y : numpy ndarray
            array of y coordinates
        xi : numpy ndarray
            array of xi values defining the x-axis range
        yi : numpy ndarray
            array of yi values defining the y-axis range
        extra_indices : tuple, optional
            Extra indices to expand the result by adding extra rows and columns.
            It is a tuple of two non-negative integers (extra_rows, extra_cols).



    Returns
    -------
        row_range : numpy ndarray
            1D array of row indices encompassing the specified yi range.
        col_range : numpy ndarray
            1D array of column indices encompassing the specified xi range.
    """
    assert np.issubdtype(x.dtype, np.number), 'X must be numeric.'
    assert x.ndim <= 2, 'This function only works for 1D or 2D X and Y arrays.'
    assert x.shape == y.shape, 'X and Y must be the same exact size.'
    assert np.issubdtype(xi.dtype, np.number), 'xi must be numeric.'
    assert np.issubdtype(yi.dtype, np.number), 'yi must be numeric.'
    
    extra_rows, extra_cols = extra_indices
    assert extra_rows >= 0, 'extrarows must be a positive integer.'
    assert extra_cols >= 0, 'extracols must be a positive integer.'

    rows_in, cols_in = x.shape

    if len(xi) == 0:
        xi = [np.min(x), np.max(x)]
    else:
        xi = [np.min(xi), np.max(xi)]

    if len(yi) == 0:
        yi = [np.min(y), np.max(y)]
    else:
        yi = [np.min(yi), np.max(yi)]

    row_i, col_i = np.where((x >= xi[0]) & (x <= xi[1]) & (y >= yi[0]) & (y <= yi[1]))

    row_range = np.arange(np.min(row_i) - 1 - extra_rows, np.max(row_i) + 2 + extra_rows)
    col_range = np.arange(np.min(col_i) - 1 - extra_cols, np.max(col_i) + 2 + extra_cols)

    row_range = row_range[(row_range >= 0) & (row_range < rows_in)]
    col_range = col_range[(col_range >= 0) & (col_range < cols_in)]

    return row_range, col_range


# In[ ]:


def geoquad_ps(ax, lat_lim, lon_lim, **kwargs):
    """
    Plots a geographic quadrangle in polar stereographic units.

    Parameters
    ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Axes on which to plot the quadrangle.
        lat_lim : list
            List containing the bounds of the shape (min and max latitude).
        lon_lim : list
            List containing the bounds of the shape (min and max latitude).


    Returns
    -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The input axes, now with the grid as well.
        """
    assert len(lat_lim) == 2 and len(lon_lim) == 2, "Error: lat_lim and lon_lim must each be two-element arrays."
    assert all(-90 <= lat <= 90 for lat in lat_lim) and all(-180 <= lon <= 180 for lon in lon_lim), "Error: lat_lim and lon_lim must be geographic coordinates."

    if np.diff(lon_lim) < 0:
        lon_lim[1] = lon_lim[1] + 360

    lat = np.concatenate((np.linspace(lat_lim[0], lat_lim[0], 200), np.linspace(lat_lim[1], lat_lim[1], 200), [lat_lim[0]]))
    lon = np.concatenate((np.linspace(lon_lim[0], lon_lim[1], 200), np.linspace(lon_lim[1], lon_lim[0], 200), [lon_lim[0]]))

    vertices = np.column_stack([lon, lat])
    path = mpath.Path(vertices)
    patch = mpatches.PathPatch(path, transform=ccrs.Geodetic(), **kwargs)
    ax.add_patch(patch)

    return ax


def ps_grid(center_x, center_y, width_km, height_km, resolution_km):
    """
    Creates a polar stereographic grid of specified spatial resolution.

    Parameters
    ----------
        center_x : int or float
            X coordinate or latitude representing the center of the grid.
        center_y : int or float
            Y coordinate or longitude representing the center of the grid.
        width_km : int or float
            The width of the grid (in kilometers).
        height_km : int or float
            The height of the grid (in kilometers).
        resolution_km : int or float
            The desired spatial resolution (in kilometers).


    Returns
    -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The input axes, now with the quadrangle.
        """
    center_x = np.array(center_x)
    center_y = np.array(center_y)

    # Convert width and resolution from km to meters
    width_m = width_km * 1000
    height_m = height_km * 1000
    resolution_m = resolution_km * 1000

    # Check that resolution is not greater than width
    assert width_m > resolution_m, "Grid width should be bigger than the grid resolution"
    assert resolution_m > 0, "Grid resolution must be greater than zero"
    assert width_m > 0, "Grid width must be greater than zero"

    if is_lat_lon(center_x, center_y):
        center_x, center_y = ll2ps(center_x, center_y)

    # Define x and y values for grid width
    x = np.arange(center_x - width_m / 2, center_x + width_m / 2, resolution_m)
    y = np.arange(center_y - height_m / 2, center_y + height_m / 2, resolution_m)

    # Create grid
    X, Y = np.meshgrid(x, y)

    # Convert coordinates if necessary
    lat, lon = ps2ll(X, Y)
    return lat, lon


def uv2vxvy(lat_or_x, lon_or_y, u, v):
    """
    Transforms georeferenced (zonal and meridional) vectors components
    to cartesian (polar stereographic) coordinate components.

    Parameters
    ----------
        lat_or_x : int or float or list or numpy ndarray
            Latitude or x-coordinates of the vector field points.
        lon_or_y : int or float or list or numpy ndarray
            Longitude or Y coordinates representing the center of the grid.
        u : int or float or list or numpy ndarray
            Zonal (eastward) component of the vector field.
        v : int or float or list or numpy ndarray
            Meridional (northward) component of the vector field.


    Returns
    -------
        vx : float or numpy ndarray
            Horizontal component of the vector field in cartesian coordinates.
        vy : float or numpy ndarray
            Vertical component of the vector field in cartesian coordinates.
        """
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
    if is_lat_lon(lat_or_x, lon_or_y):
        lon = lon_or_y  # lat is really just a placeholder to make the function a little more intuitive to use. It is not necessary for calculation.
    else:
        _, lon = ps2ll(lat_or_x, lon_or_y)

    # Convert lon to radians
    lon_rad = np.deg2rad(lon)

    # Perform calculation
    vx = u * np.cos(lon_rad) + v * np.sin(lon_rad)
    vy = -u * np.sin(lon_rad) + v * np.cos(lon_rad)

    return vx, vy


def path_dist(lat, lon, units='m', ref_point=None):
    """
    Calculates cumulative distance traveled along a path given by the arrays lat and lon.

    Parameters
    ----------
        lat : list or numpy ndarray
            Latitude coordinates of the traveled path.
        lon : list or numpy ndarray
            Longitude coordinates of the traveled path.
        units : string
            Specifies the distance metric to use. Options are meters ('m'),
            kilometers ('km'), nautical miles ('nm'), feet ('ft'),
            inches ('in'), yards ('yd'), and miles ('mi').
        ref_point: int or float or list or numpy ndarray
            Meridional (northward) component of the vector field.


    Returns
    -------
        path_distance : float or numpy ndarray
            Distance(s) traveled along the given path(s).
        """
    assert len(lat) == len(lon), 'Length of lat and lon must match.'
    assert len(lat) > 1, 'lat and lon must have more than one point.'

    # Check if reference point is defined:
    if ref_point is not None:
        assert len(
            ref_point) == 2, 'Coordinates of reference point can be only a single point given by a latitude/longitude pair in the form [reflat reflon].'

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
        if ref_point is not None and (lat[i - 1], lon[i - 1]) == tuple(ref_point):
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


def in_ps_quad(lat, lon, lat_lim, lon_lim):
    """
    Returns true for points inside a polar stereographic quadrangle.

    Parameters
    ----------
        lat : numpy ndarray
            Latitude coordinate(s) of the desired point(s).
        lon : numpy ndarray
            Longitude coordinates of the desired point(s).
        lat_lim : numpy ndarray
            Array containing the bounds of the shape (min and max latitude).
        lon_lim: numpy ndarray
            List containing the bounds of the shape (min and max longitude).


    Returns
    -------
        IN : bool
            Returns True if the point is inside the quadrangle.
        """
    assert np.array(lat.shape) == np.array(lon.shape), 'Inputs lat and lon must be the same size.'
    assert np.array(lat_lim.shape) == np.array(
        lon_lim.shape), 'Inputs lat_lim_or_xlim and lon_lim_or_ylim must be the same size.'
    assert len(lat_lim) > 1, 'lat_lim or xlim must have more than one point.'

    min_lat, max_lat = min(lat_lim), max(lat_lim)
    min_lon, max_lon = min(lon_lim), max(lon_lim)

    IN = np.logical_and(lat >= min_lat, lat <= max_lat)
    IN = np.logical_and(IN, lon >= min_lon)
    IN = np.logical_and(IN, lon <= max_lon)

    return IN


def ps_distortion(lat, true_lat=-71):
    """
    Approximates the map scale factor for a polar stereographic projection.
    It is the ratio of distance on a ps projection to distance on a sphere.

    Parameters
    ----------
        lat : int or float or list or numpy ndarray
            Latitude(s) at which the map scale factor is calculated.
        true_lat : int or float, optional
            Latitude of true scale for the polar stereographic projection.
            Default is -71 degrees.

    Returns
    -------
        m : float
            The calculated map scale factor.
        """
    assert np.all(np.abs(lat) <= 90), 'Error: inputs must be latitudes.'
    assert np.isscalar(true_lat), 'Error: true_lat must be a scalar.'
    assert np.abs(true_lat) <= 90, 'Error: true_lat must be in the range -90 to 90.'

    lat = np.radians(lat)  # convert from degrees to radians
    true_lat = np.radians(true_lat)  # same for true_lat

    # calculate map scale factor
    m = (1 + np.sin(np.abs(true_lat))) / (1 + np.sin(np.abs(lat)))
    return m


def path_dist_ps(lat_or_x, lon_or_y, *args):
    """
        Returns the cumulative distance along a path in polar stereographic coordinates.

        Parameters
        ----------
            lat_or_x : numpy ndarray
                Latitude or x-coordinates of the traveled path.
            lon_or_y : numpy ndarray
                Longitude or y-coordinates of the traveled path.

        Returns
        -------
            d : numpy ndarray
                Distance(s) traveled along the given path(s) in polar stereographic coordinates..
            """
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
    if is_lat_lon(lat_or_x, lon_or_y):
        lat = lat_or_x
        [x, y] = ll2ps(lat_or_x, lon_or_y)
    else:
        x = lat_or_x
        y = lon_or_y
        lat, _ = ps2ll(x, y)  # don't need lon

    # Perform mathematics:
    m = ps_distortion(lat[1:])  # Assuming ps_distortion is defined or imported

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


def ps_path(lat_or_x, lon_or_y, spacing, method='linear'):
    """
    Interpolates a path in polar stereographic coordinates to generate equally spaced coordinates along the path.

    Parameters
    ----------
    lat_or_x : numpy ndarray
        Latitude or x-coordinates of the input path.
    lon_or_y : numpy ndarray
        Longitude or y-coordinates of the input path.
    spacing : scalar
        Spacing between the generated coordinates along the path.
    method : str, optional
        Interpolation method to be used (default is 'linear').

    Returns
    -------
    out1 : numpy ndarray
        Interpolated x or latitude coordinates along the path.
    out2 : numpy ndarray
        Interpolated y or longitude coordinates along the path.
    """
    assert isinstance(lat_or_x, np.ndarray) and lat_or_x.ndim == 1, 'Input error: input coordinates must be vectors of matching dimensions.'
    assert lat_or_x.shape == lon_or_y.shape, 'Input error: dimensions of input coordinates must match.'
    assert np.isscalar(spacing), 'Input error: spacing must be a scalar.'

    geoin = is_lat_lon(lat_or_x, lon_or_y)
    if geoin:
        x, y = ll2ps(lat_or_x, lon_or_y)
    else:
        x = lat_or_x
        y = lon_or_y

    d = path_dist_ps(x, y)

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


def path_crossing_ps71(lat_1, lon_1, lat_2, lon_2, clip_option=None):
    """
        Finds the intersection point(s) of two paths in polar stereographic coordinates with a standard parallel at 71°S.

        Parameters
        ----------
        lat_1 : list of floats
            Latitude coordinates of the first path.
        lon_1 : list of floats
            Longitude coordinates of the first path.
        lat_2 : list of floats
            Latitude coordinates of the second path.
        lon_2 : list of floats
            Longitude coordinates of the second path.
        clip_option : str, optional
            Option to clip outliers during computation ('on' or 'off', default is None).

        Returns
        -------
        lat_i : numpy ndarray
            Latitude coordinate(s) of the intersection point(s).
        lon_i : numpy ndarray
            Longitude coordinate(s) of the intersection point(s).
        """

    assert isinstance(lat_1, list) and all(isinstance(i, float) for i in lat_1), 'Input lat1 must be a list of floats.'
    assert len(lat_1) == len(lon_1), 'Input lat1 and lon1 must be the same size.'
    assert isinstance(lat_2, list) and all(isinstance(i, float) for i in lat_2), 'Input lat2 must be a list of floats.'
    assert len(lat_2) == len(lon_2), 'Input lat2 and lon2 must be the same size.'

    clip_data = True

    if clip_option is not None:
        if clip_option.lower().startswith('no') or clip_option.lower() == 'off':
            clip_data = False

    # Transform to polar stereo coordinates with standard parallel at 71 S
    # Here we assume that ll2ps and ps2ll are already defined functions
    x1, y1 = ll2ps(lat_1, lon_1)
    x2, y2 = ll2ps(lat_2, lon_2)

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
    line1 = LineString(np.column_stack([x1, y1]))
    line2 = LineString(np.column_stack([x2, y2]))
    intersection = line1.intersection(line2)

    if not intersection.is_empty:
        if intersection.geom_type == 'MultiPoint':
            # Extract coordinates from MultiPoint
            x, y = intersection.xy
            intersections = np.column_stack((x, y))
        elif intersection.geom_type == 'Point':
            # Single intersection point
            intersections = np.array([intersection.xy])
        elif intersection.geom_type == 'LineString':
            # Single LineString intersection
            intersections = np.array([intersection.coords])
        else:
            # Handle other intersection geometries (e.g., MultiLineString)
            intersections = [np.array(line.coords) for line in intersection.geoms if line.geom_type == 'LineString']
            intersections = np.concatenate(intersections)

        # Transform back to lat/lon space
        lat_i, lon_i = ps2ll(intersections[:, 0], intersections[:, 1])
        return lat_i, lon_i

    # If no intersections are found, return None
    return None


def clip_outliers(x, y, x_compare, std, axis):
    """
    Clips outliers from the input coordinates based on the specified axis and standard deviation.

    Parameters
    ----------
    x : float or list of floats
        x coordinate(s) to be clipped.
    y : float or list of floats
        y coordinate(s) to be clipped.
    x_compare : float or list of floats
        Reference x or latitude coordinate(s) for mean comparison.
    std : float or int
        Number of standard deviations for the clipping threshold.
    axis : string
        Axis along which outliers will be clipped ('x' or 'y').

    Returns
    -------
    x : numpy ndarray
        Clipped x or latitude coordinates.
    y : numpy ndarray
        Clipped y or longitude coordinates.
    """
    if axis == 'x':
        y = y[np.abs(x - np.mean(x_compare)) < std]
        x = x[np.abs(x - np.mean(x_compare)) < std]
    elif axis == 'y':
        x = x[np.abs(y - np.mean(x_compare)) < std]
        y = y[np.abs(y - np.mean(x_compare)) < std]
    return x, y


def inter_x(L1, L2):
    """
    Finds the intersection point(s) between two paths represented by coordinates.

    Parameters
    ----------
    L1 : list of numpy ndarrays
        Coordinates of the first path.
    L2 : list of numpy ndarrays
        Coordinates of the second path.

    Returns
    -------
    intersection : list or None
        List containing intersection point(s) coordinates [x, y] or None if no intersection.
    """
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


def ant_bounds(show_ticks=False):
    """
    Creates a polar stereographic map of the Antarctic region with coastlines and ice shelves.

    Parameters
        ----------
        show_ticks : bool, optional
            Adds ticks and axes to the plot.

    Returns
    -------
    ax : GeoAxesSubplot
        Matplotlib GeoAxesSubplot object with the Antarctic map.
    """

    # Use the relative path based on the current script's directory
    data_dir = files("PyPMT.data")

    # Read shapefiles from the data folder
    moa_coast = gpd.read_file(data_dir / 'moa2014_coastline_v01.shp')
    moa_gl = gpd.read_file(data_dir / 'moa2014_grounding_line_v01.shp')
    moa_islands = gpd.read_file(data_dir / 'moa2014_islands_v01.shp')
    polar = ccrs.SouthPolarStereo(true_scale_latitude=-71)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection=polar))
    ax.add_geometries(moa_coast.geometry, ccrs.SouthPolarStereo(true_scale_latitude=-71), facecolor='lightblue',
                      edgecolor='k', zorder=1)
    ax.add_geometries(moa_gl.geometry, ccrs.SouthPolarStereo(true_scale_latitude=-71), facecolor='white',
                      edgecolor='k', zorder=1)
    ax.add_geometries(moa_islands.geometry, ccrs.SouthPolarStereo(true_scale_latitude=-71), facecolor='white',
                      edgecolor='k', zorder=1)
    ax.set_xlim([-3.3e6, 3.3e6])
    ax.set_ylim([-3.3e6, 3.3e6])

    # Add ticks if show_ticks is True
    if show_ticks:
        ax.set_xticks([-3e6, -1.5e6, 0, 1.5e6, 3e6], crs=polar)
        ax.set_yticks([-3e6, -1.5e6, 0, 1.5e6, 3e6], crs=polar)
        # Set axis labels
        ax.set_xlabel("Easting (m)", labelpad=15)
        ax.set_ylabel("Northing (m)", labelpad=15)

    return ax


def plot_ps(ax, lat, lon, km=False, **kwargs):
    """
    Plots points in polar stereographic coordinates on a specified GeoAxesSubplot.

    Parameters
    ----------
    ax : GeoAxesSubplot
        Matplotlib GeoAxesSubplot object where the points will be plotted.
    lat : float or list of floats
        Latitude coordinate(s) of the point(s) to be plotted.
    lon : float or list of floats
        Longitude coordinate(s) of the point(s) to be plotted.
    km : bool, optional
        If True, coordinates are assumed to be in kilometers (default is False).
    
    Returns
    -------
    lines : list
        List of Line2D objects representing the plotted points.
    """

    if np.isscalar(lat):
        lat = [lat]
    if np.isscalar(lon):
        lon = [lon]
    x, y = ll2ps(lat, lon)
    if km:
        x, y = x / 1000, y / 1000
    return ax.plot(x, y, **kwargs)


def pcolor_ps(ax, x, y, z, **kwargs):
    """
    Creates a pseudocolor plot in polar stereographic coordinates on a specified GeoAxesSubplot.

    Parameters
    ----------
    ax : GeoAxesSubplot
        Matplotlib GeoAxesSubplot object where the pseudocolor plot will be created.
    x : array-like
        x or latitude coordinates of the grid.
    y : array-like
        y or longitude coordinates of the grid.
    z : array-like
        Values for the pseudocolor plot.
        
    Returns
    -------
    h : QuadMesh
        Matplotlib QuadMesh object representing the pseudocolor plot.
    """
    if len([x, y, z]) < 3:
        raise ValueError('The pcolor_ps function requires at least three inputs: x, y, and Z.')

    if not np.issubdtype(np.array(x).dtype, np.number):
        raise ValueError('pcolor_ps requires numeric inputs for x.')

    if not np.issubdtype(np.array(y).dtype, np.number):
        raise ValueError('pcolor_ps requires numeric inputs for y.')

    # Create a pseudo color plot
    h = ax.pcolormesh(x, y, z, **kwargs)

    return h


def plot_3ps(lat, lon, z, x, y, extra_m=50e3, z_scale=1.0, show_grid=True, **kwargs):
    """
        Creates and displays three-dimensional plots in polar stereographic coordinates.

        Parameters
        ----------
        lat : int or float or list or tuple or numpy ndarray
            Latitude coordinate(s) representing the geographic center of the data.
        lon : int or float or list or tuple or numpy ndarray
            Longitude coordinate(s) representing the geographic center of the data.
        z : numpy ndarray
            z coordinates to plot.
        x : numpy ndarray
            x coordinates to plot.
        y : numpy ndarray
            y coordinates to plot.
        extra_m :
            The maximum distance (meters) between the plotted x and y coordinates and the
            input latitude / longitude coordinates.
        z_scale : int or float
            The scaling factor to use for the z values.
        show_grid : boolean
            Default is True to plot on a grid. False turns off the grid.
        Returns
        -------
        None
        """
    # Check if lat and lon are single values or lists/arrays
    if not isinstance(lat, (list, tuple, np.ndarray)):
        lat = [lat]
        lon = [lon]
    else:
        assert len(lat) == len(lon), "The number of latitude and longitude values should be the same."
    assert isinstance(lat, (int, float, list, tuple, np.ndarray)), "plot_3ps requires numeric inputs first."
    assert isinstance(lon, (int, float, list, tuple, np.ndarray)), "plot_3ps requires numeric inputs first."

    psx, psy = ll2ps(lat, lon, **kwargs)  # Assume ll2ps function is available

    # Create masks to limit the plot area
    maskx = np.abs(x - psx) < extra_m
    masky = np.abs(y - psy) < extra_m

    # Create a 2D mask from the x and y masks
    mask2d = np.outer(maskx, masky)

    # Apply the 2D mask to the z values
    z_mask = z[mask2d]

    # Create masked x and y arrays
    x_msk = x[maskx]
    y_msk = y[masky]

    # Create a meshgrid from the masked x and y arrays
    xx_msk, yy_msk = np.meshgrid(x_msk, y_msk)
    z_masked = z_mask.reshape((x_msk.shape[0], y_msk.shape[0]))

    # Create a new figure and 3D subplot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    cmap = kwargs.get('cmap')

    if cmap is not None:
        # Plot the 3D surface with scaled z-values for visual effect
        surf = ax.plot_surface(xx_msk, yy_msk, z_masked * z_scale, **kwargs)

        # Set z-axis limits based on the true (unscaled) z values
        z_min, z_max = np.min(z_masked), np.max(z_masked)
        ax.set_zlim(z_min, z_max)

        # Create custom z-ticks to show true values
        num_ticks = 5  # You can adjust this number as needed
        z_ticks = np.linspace(z_min, z_max, num_ticks)
        ax.set_zticks(z_ticks * z_scale)
        ax.set_zticklabels([f'{z:.0f}' for z in z_ticks])

        # Add a colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('True Z Value')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (True Scale)')

        # Control grid visibility
        ax.grid(show_grid)

        # Add a text annotation explaining the visual scaling
        ax.text2D(0.05, 0.95, f'Visual Z-Scale: {z_scale}', transform=ax.transAxes)
    else:
        # Plot the 3D surface with scaled z-values for visual effect
        surf = ax.plot_surface(xx_msk, yy_msk, z_masked * z_scale, **kwargs)

        # Set z-axis limits based on the true (unscaled) z values
        z_min, z_max = np.min(z_masked), np.max(z_masked)
        ax.set_zlim(z_min, z_max)

        # Create custom z-ticks to show true values
        num_ticks = 5  # You can adjust this number as needed
        z_ticks = np.linspace(z_min, z_max, num_ticks)
        ax.set_zticks(z_ticks * z_scale)
        ax.set_zticklabels([f'{z:.0f}' for z in z_ticks])

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (True Scale)')

        # Control grid visibility
        ax.grid(show_grid)

        # Add a text annotation explaining the visual scaling
        ax.text2D(0.05, 0.95, f'Visual Z-Scale: {z_scale}', transform=ax.transAxes)

    plt.show()


def circle_ps(ax, lons, lats, radii, km=False, **kwargs):
    """
        Plots circles (in polar stereographic coordinates) of given radii on a given plot.

        Parameters
        ----------
        ax : GeoAxesSubplot
            Matplotlib GeoAxesSubplot object where the points will be plotted.
        lons : int or float or list or numpy ndarray
            Longitude coordinate(s) for the center of circle(s).
        lats : int or float or list or numpy ndarray
            Latitude coordinate(s) for the center of circle(s).
        radii : int or float or list or numpy ndarray
            Radius or radii for circle(s). Default is in meters.
        km : bool
            True when radii are measured in kilometers, False (default) when
            radii are measured in meters.
        Returns
        -------
        None
        """
    geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))

    if km:
        radii = radii * 1000

    if not isinstance(lons, np.ndarray):
        lons = np.array([lons])
    if not isinstance(lats, np.ndarray):
        lats = np.array([lats])
    if not isinstance(radii, np.ndarray):
        radii = np.array([radii])

    assert len(lons) == len(lats) == len(radii), 'Input arrays must have the same length'

    for lon, lat, radius in zip(lons, lats, radii):
        geod = Geodesic()
        circle_points = geod.circle(lon, lat, radius, n_samples=100)
        ax.plot(circle_points[:, 0], circle_points[:, 1], transform=geodetic, **kwargs)


def patch_ps(ax, lat, lon, *args, **kwargs):
    """
        Description:
        ------------
        Plot a polygon patch on a given map using Matplotlib's GeoAxesSubplot.

        Parameters:
        -----------
        ax : GeoAxesSubplot GeoAxesSubplot object where the patch will be added.
        lat : array-like latitude coordinates of the polygon vertices.
        lon : array-like longitude coordinates of the polygon vertices.

        Returns:
        --------
        ax : GeoAxesSubplot GeoAxesSubplot object with the added patch.

        """
    # Input checks
    assert len(lat) > 1 and len(lon) > 1, 'The patchps function requires at least two input: lat and lon.'
    assert isinstance(lat, (list, np.ndarray)) and isinstance(lon, (list, np.ndarray)), 'patchps requires numeric inputs first.'
    assert np.abs(lat).max() <= 90, 'Some of your latitudes have absolute values exceeding 90 degrees.'

    # Parse inputs
    plotkm = False  # By default, plot in meters
    meridian = 0    # Top of the map is Fimbul Ice Shelf

    if 'km' in kwargs:
        plotkm = kwargs['km']
        del kwargs['km']

    if 'meridian' in kwargs:
        meridian = kwargs['meridian']
        del kwargs['meridian']
        assert np.isscalar(meridian), 'Error: meridian must be a scalar longitude.'

    # Convert units and plot
    x, y = ll2ps(lat, lon, meridian=meridian)

    # Convert to kilometers if user requested
    if plotkm:
        x /= 1000
        y /= 1000

    h = ax.fill(x, y, *args, **kwargs)

    return h


def scar_label(ax, feature_name, *args, **kwargs):
    """
    Labels Antarctic features on a map. Feature names and locations correspond to
    25,601 locations identified by the Scientific Committee on Antarctic Research (SCAR).

    Parameters
    ----------
    ax : GeoAxesSubplot
        Matplotlib GeoAxesSubplot object where the features will be labeled.
    feature_name : string or list of strings
        Name(s) of the feature(s) to be labeled.
    Returns
    -------
    ax : GeoAxesSubplot
        Returns the Matplotlib GeoAxesSubplot with the features labeled.
    """
    # Check if at least one input is provided
    assert len(feature_name) > 0, "The scar_label requires at least one input. What are you trying to label?"

    # Check if the feature_name is a string or a list of strings
    if isinstance(feature_name, str):
        feature_name = [feature_name]
    elif isinstance(feature_name, list):
        assert all(isinstance(name, str) for name in feature_name), "Feature names must be strings."
    else:
        raise ValueError("Feature names must be a string or a list of strings.")

    plot_marker = False
    if 'marker' in args:
        plot_marker = True
        marker_style = args[args.index('marker') + 1]
        args.remove('marker')

    feature_lat, feature_lon = [], []
    for name in feature_name:
        if name.lower() == 'glaciers':
            data = np.load('scarnames.npz')
            lat = data['lat']
            lon = data['lon']
            names = data['names']
            feature_Type = data['featureType']
            ind = np.where(feature_type == 'glacier')[0]
            feature_lat.extend(lat[ind])
            feature_lon.extend(lon[ind])
        elif name.lower() == 'ice shelves':
            data = np.load('scarnames.npz')
            lat = data['lat']
            lon = data['lon']
            names = data['names']
            feature_type = data['featureType']
            ind = np.where(feature_type == 'ice shelf')[0]
            feature_lat.extend(lat[ind])
            feature_lon.extend(lon[ind])
        else:
            lat, lon = scar_loc(name)
            feature_lat.append(lat)
            feature_lon.append(lon)

    # Place text label
    for i in range(len(feature_name)):
        ax.text(feature_lon[i], feature_lat[i], feature_name[i], horizontalalignment='center', verticalalignment='top', transform=ccrs.PlateCarree(), **kwargs)

    # Format text and marker
    for i in range(len(args)):
        if args[i] != 'marker':
            ax.setp(ax.gca().texts[i], **{args[i]: args[i + 1]})

    if plot_marker:
        ax.scatter(feature_lon, feature_lat, marker=marker_style, transform=ccrs.PlateCarree())

    return ax


def ps2wkt(lati_or_xi, loni_or_yi, filename=None):
    """
    Converts polar stereographic coordinates to WKT format.

    Parameters
    ----------
    lati_or_xi : array-like
        1D array of latitude or polar stereographic x-coordinates.
    loni_or_yi : array-like
        1D array of longitude or polar stereographic y-coordinates.
    filename : str or None, optional
        If specified, write WKT to a file with this name.

    Returns
    -------
    str or None
        If `filename` is not specified, the WKT string is returned. Otherwise, `None` is returned.

    """
    assert len(lati_or_xi) == len(loni_or_yi), 'Error: Dimensions of input coordinates do not agree.'
    if isinstance(lati_or_xi[0], float) or isinstance(lati_or_xi[0], int):
        lati, loni = ps2ll(lati_or_xi, loni_or_yi)
        return_wkt = True
    else:
        return_wkt = False
        lati, loni = lati_or_xi, loni_or_yi  # Assign input arguments to new variables

    # Combine loni and lati into a matrix
    coords = np.column_stack((loni, lati))

    # convert the matrix to a string
    coords_str = np.array2string(coords, formatter={'all': lambda x: str(x)})

    # Replace square brackets with parentheses
    coords_str = coords_str.replace('[', '(').replace(']', ')')

    # Replace semicolons with commas and spaces
    coords_str = coords_str.replace(';', ', ')

    # Use regular expressions to remove consecutive "(, )" sequences
    coords_str = re.sub(r'\(, +\)', '), (', coords_str)

    # Create the WKT string with "POLYGON" prefix
    wkt = f'POLYGON({coords_str})'

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(wkt)
            return None
    if return_wkt:
        return wkt
    else:
        return np.array(lati), np.array(loni)


def scatter_ps(ax, lat, lon, s=100, c='b', **kwargs):
    """
    Plots georeferenced data in polar stereographic coordinates.

    Parameters
    ----------
    ax : GeoAxesSubplot
        Matplotlib GeoAxesSubplot object where the data will be plotted.
    lat : int or float or list of int or list of float
        Latitude coordinate(s) of the data.
    lon : int or float or list of int or list of float
        Longitude coordinate(s) of the data.
    s : int or float
        Size of the data points.
    c : string
        Color of the plotted data points.
    Returns
    -------
    ax : GeoAxesSubplot
        Returns the Matplotlib GeoAxesSubplot with the data points plotted.
    """
    if not isinstance(lat, (list, tuple, np.ndarray)):
        lat = [lat]
        lon = [lon]
    else:
        assert len(lat) == len(lon), "The number of latitude and longitude values should be the same."
    assert isinstance(lat, (int, float, list, tuple, np.ndarray)), "scatter_ps requires numeric inputs first."
    assert isinstance(lon, (int, float, list, tuple, np.ndarray)), "scatter_ps requires numeric inputs first."
    assert np.max(np.abs(lat)) <= 90, "I suspect you have entered silly data into plot_ps because some of your latitudes have absolute values exceeding 90 degrees."

    # Plot the points on the map
    ax.scatter(lon, lat, s=s, c=c, transform=ccrs.PlateCarree(), **kwargs)


def greenland_bounds():
    """
    Creates a polar stereographic map of the Greenland region with coastlines.

    Returns
    -------
    ax : Matplotlib GeoAxesSubplot
        Matplotlib GeoAxesSubplot object with the Antarctic map.
    """
    fig = plt.figure(figsize=(10, 10))
    map_proj = ccrs.Orthographic(central_longitude=-45.0, central_latitude=70.0)
    ax = fig.add_subplot(1, 1, 1, projection=map_proj)
    ax.set_extent([-74, -11, 59, 83], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    return ax


def graticule_ps(lats=None, lons=None, clipping=False, ax=None, **kwargs):
    """
        Places a graticule on a polar stereographic cartesian coordinate map.

        Parameters
        ----------
        lats : list, optional
            Specifies lines of latitude.
        lons : list, optional
            Specifies lines of longitude.
        clipping : bool, optional
            True deletes all graticule points outside the extent of the current plot.
        ax : GeoAxesSubplot, optional
            GeoAxesSubplot to place the graticule on.
        Returns
        -------
        None
        """

    if lats is None:
        lats = [-85] + list(range(-80, -20, 10))
    if lons is None:
        lons = list(range(-150, 210, 30))

    num_pts = 360  # 360 points per line

    # create lat/lon points of all lines
    lat = np.empty(len(lats) * num_pts + len(lats) + len(lons) * num_pts + len(lons) - 1, dtype=float)
    lat.fill(np.nan)

    lon = np.empty(len(lats) * num_pts + len(lats) + len(lons) * num_pts + len(lons) - 1, dtype=float)
    lon.fill(np.nan)

    n = 0

    for k in range(len(lats)):
        lat[n:n + num_pts] = lats[k]
        lon[n:n + num_pts] = np.linspace(-180, 180, num_pts)
        n += num_pts + 1

    for k in range(len(lons)):
        lon[n:n + num_pts] = lons[k]
        lat[n:n + num_pts] = np.linspace(np.min(lats), np.max(lats), num_pts)
        n += num_pts + 1

    # Get current axis limits
    if ax is None:
        ax = plt.gca()
    ax_limits = ax.axis()

    # Check if a map was already open
    if ax_limits != [0, 1, 0, 1]:
        # If user requests polar stereographic kilometers, we have to convert current axis limits to km for comparison:
        if 'km' in kwargs:
            ax_limits = ax_limits * 1000
        # If clipping is true, clip the graticule to the current axis limits
        if clipping:
            # Find indices of lat, lon inside current axis limits
            ind = np.where(
                (lat >= ax_limits[0]) & (lat <= ax_limits[1]) & (lon >= ax_limits[2]) & (lon <= ax_limits[3]))
            # Set everything outside current axis limits to NaN
            lat[~np.isin(np.arange(len(lat)), ind)] = np.nan
            lon[~np.isin(np.arange(len(lon)), ind)] = np.nan

    plot_ps(ax, lat, lon, color='gray')


def text_ps(lat, lon, string, plot_km=False, **kwargs):
    """
            Places georeferenced text labels in polar stereographic cartesian coordinates

            Parameters
            ----------
            lat : int, float
                Specifies line of latitude.
            lon : int, float
                Specifies line of longitude.
            string : string
                The desired text to place on the plot.
            plot_km : bool, optional

            Returns
            -------
            h : Matplotlib GeoAxesSubplot
            Matplotlib GeoAxesSubplot object with the added text.
            """
    # Convert lat, lon to polar stereographic coordinates
    x, y = ll2ps(lat, lon)

    # Convert to kilometers if the user requests it
    if plot_km:
        x /= 1000
        y /= 1000

    # Add the text label at the specified coordinates
    h = plt.text(x, y, string, horizontalalignment='center', **kwargs)
    return h
