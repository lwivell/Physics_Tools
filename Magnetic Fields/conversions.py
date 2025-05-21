from typing import Tuple
import numpy as np

def spherical_unit_vectors(pos_theta, pos_phi):   

    """
    Calculates the spherical unit vectors for magnetic field component transformation

    Parameters
    -------------
    pos_theta
        1D array of the theta component of position
    pos_phi
        1D array of the phi component of position

    Returns
    unitr
        unit vector in i, for the radial component
    unittheta
        unit vector in j, for the theta component
    unitphi
        unit vector in k, for the phi component
    """
    unitr = np.array((np.sin(pos_theta)*np.cos(pos_phi), np.sin(pos_theta)*np.sin(pos_phi), np.cos(pos_theta)))
    unittheta = np.array((np.cos(pos_theta)*np.cos(pos_phi), np.cos(pos_theta)*np.sin(pos_phi), -np.sin(pos_theta)))
    unitphi = np.array((-np.sin(pos_phi), np.cos(pos_phi), np.zeros_like(pos_phi)))
    return unitr, unittheta, unitphi

def cartesian_to_spherical(posx, posy, posz):
    """
    Converts positions in spherical coordinates to cartesian coordinates
    
    Parameters
    ------------
    pos_x
        1D array of the x position
    pos_y
        1D array of the y position
    pos_z
        1D array of the z position  

    Returns
    -------------
    pos_r
        1D array of the radial component of position
    pos_theta
        1D array of the theta component of position
    pos_phi
        1D array of the phi component of position
    
    """
    posr = np.sqrt((posx**2)+(posy**2)+(posz**2))
    postheta = np.arccos(posz/(np.sqrt((posx**2)+(posy**2)+(posz**2))))
    posphi = np.sign(posy)*np.arccos(posx/(np.sqrt((posx**2)+(posy**2)))) 
    return posr, postheta, posphi

def spherical_to_cartesian_pos(pos_r, pos_theta, pos_phi):
    """
    Converts positions in spherical coordinates to cartesian coordinates
    
    Parameters
    ------------
    pos_t
        1D array of the times at which positions were measured
    pos_r
        1D array of the radial component of position
    pos_theta
        1D array of the theta component of position
    pos_phi
        1D array of the phi component of position

    Returns
    -------------
    pos_x
        1D array of the x position
    pos_y
        1D array of the y position
    pos_z
        1D array of the z position
    
    """
    pos_x = pos_r * np.cos(pos_phi) * np.sin(pos_theta)  #shifts are negative as it moves measurement point not centre itself
    pos_y = pos_r * np.sin(pos_phi) * np.sin(pos_theta)
    pos_z = pos_r * np.cos(pos_theta)
    return pos_x, pos_y, pos_z


def spherical_to_cartesian_MAG(B_r, B_theta, B_phi, colat, long):
    """
    Convert magnetic field components from spherical to Cartesian coordinates.

    Parameters:
    B_r, B_theta, B_phi : array-like
        Magnetic field components in spherical coordinates.
    colat, long : array-like
        Spherical coordinate angles in radians.

    Returns:
    B_x, B_y, B_z : array-like
        Magnetic field components in Cartesian coordinates.
    """
    B_x = B_r * np.sin(colat) * np.cos(long) + B_theta * np.cos(colat) * np.cos(long) - B_phi * np.sin(long)
    B_y = B_r * np.sin(colat) * np.sin(long) + B_theta * np.cos(colat) * np.sin(long) + B_phi * np.cos(long)
    B_z = B_r * np.cos(colat) - B_theta * np.sin(colat)

    return B_x, B_y, B_z