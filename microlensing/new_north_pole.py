"""

This module provides a small set of utility functions for transforming celestial coordinates
(Right Ascension and Declination) into a new spherical coordinate system defined by a different
north pole orientation.
"""
import numpy as np

def great_circle_distance(phi1, phi2, dlambda):
    """
    Computes the great circle distance between two points on a sphere.
    """
    cos_sigma = (
        np.sin(phi1) * np.sin(phi2) +
        np.cos(phi1) * np.cos(phi2) * np.cos(dlambda)
    )
    result = np.arccos(np.clip(cos_sigma, -1, 1))  # Clip to avoid floating point issues
    return result

def spherical_law_of_cosines(a, b, c):
    """
    Using the arc lengths of a triangle on a sphere (a, b, c),
    computes the angle opposite to side c using the spherical law of cosines
    """
    cos_angle = (
        (np.cos(c) - np.cos(a) * np.cos(b)) /
        (np.sin(a) * np.sin(b))
    )
    result = np.arccos(np.clip(cos_angle, -1, 1))  # Clip to avoid floating point issues
    return result

def ra_dec_to_new_north_pole(
        ra_dec, new_north_pole=(80.89166667, -69.75666667)
    ):
    """
    Convert RA/Dec coordinates to a new coordinate system 
    with a new north pole. By default, the new north pole is the
    center of the Large Magellanic Cloud.

    Parameters:
        ra_dec (array): 
           Shape (n, 2) array of ra and dec coordinates in degrees.
           First column ra, second column dec.
        kwargs:
            new_north_pole (tuple (float, float)):
                Tuple containing ra and dec of new north pole. 
                Default: (80.89166667, -69.75666667)
    
    Returns:
        result (array):
            Shape (n, 2) of transformed coordinates in  new north pole
            coordinate system in degrees.
    """
    # Convert degrees to radians
    ra = np.radians(ra_dec[:, 0])
    dec = np.radians(ra_dec[:, 1])
    ra_new_np = np.radians(new_north_pole[0])
    dec_new_np = np.radians(new_north_pole[1])
 
    # Compute angular separation (rho) between (ra, dec) and
    # the new north pole (lmc (ra, dec) by default)
    # using the great circle distance
    rho = great_circle_distance(dec, dec_new_np, ra - ra_new_np)
 
    # Compute position angle (phi)
    # phi = 0 points towards old north pole.
    phi = (
        spherical_law_of_cosines(rho, np.pi / 2 - dec_new_np, np.pi / 2 - dec)
        * np.sign(ra - ra_new_np)
    )
    result = np.degrees(np.column_stack((rho, phi)))
    return result
