from typing import Tuple
import numpy as np
from .conversions import spherical_to_cartesian_pos
from .conversions import spherical_unit_vectors

def GeoMag(pos_t, pos_r, pos_theta, pos_phi, DIPtheta, DIPphi, x_shift=0.0, y_shift=0.0, z_shift=0.0):
    """
    Transforms a Geographic centred and aligned coordinate system into an OTD centred and aligned system

    Parameters
    -----------------------------------------
    pos_t
        1D array of the times at which positions were measured
    pos_r
        1D array of the radial component of position
    pos_theta
        1D array of the theta component of position
    pos_phi
        1D array of the phi component of position
    DIPtheta
        theta tilt of the dipole
    DIPphi
        phi tilt of the dipole
    x_shift
        dipole centre shift in the x direction
    y_shift
        dipole centre shift in the y direction
    z_shift
        dipole centre shift in the z direction

    Returns
    pos_t
        1D array of the times at which positions were measured
    magposr
        1D array of radial positions
    magpostheta
        1D array of colatitude component
    magposphi
        1D array of azimuthal component 
    """

    posx, posy, posz = spherical_to_cartesian_pos(pos_r, pos_theta, pos_phi)
    
    lambdaD = np.deg2rad(DIPphi) 
    phiD = (np.pi/2) - np.deg2rad(DIPtheta)

    omega = lambdaD + (np.pi/2)
    theta = (np.pi/2) - phiD
    phi = -(np.pi/2)
    
    
    a = (np.cos(phi)*np.cos(omega)) - (np.sin(phi)*np.sin(omega)*np.cos(theta))
    b = (np.cos(phi)*np.sin(omega)) + (np.sin(phi)*np.cos(omega)*np.cos(theta))
    c = np.sin(phi)*np.sin(theta)
    d = (-np.sin(phi)*np.cos(omega)) - (np.cos(phi)*np.sin(omega)*np.cos(theta))
    e = (-np.sin(phi)*np.sin(omega)) + (np.cos(phi)*np.cos(omega)*np.cos(theta))
    f = np.cos(phi)*np.sin(theta)
    g = np.sin(omega)*np.sin(theta)
    h = -np.cos(omega)*np.sin(theta)
    i = np.cos(theta)

    RotMat = np.array([[a,b,c],[d,e,f],[g,h,i]])
    geopos = np.vstack((posx, posy, posz))

    RotPos = np.matmul(RotMat, geopos)

    magposx = RotPos[0] - x_shift
    magposy = RotPos[1] - y_shift
    magposz = RotPos[2] - z_shift

    magposr = np.sqrt((magposx**2)+(magposy**2)+(magposz**2))
    magpostheta = np.arccos(magposz/(np.sqrt((magposx**2)+(magposy**2)+(magposz**2))))
    magposphi = np.sign(posy)*np.arccos(posx/(np.sqrt((posx**2)+(posy**2))))
    
    return pos_t, magposr, magpostheta, magposphi, RotPos