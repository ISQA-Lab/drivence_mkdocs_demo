from typing import List

from astropy import coordinates
from scipy.spatial.transform import Rotation as RR
import numpy as np
import math

def is_right_hand_system(x, y, z):
    """Return True if the basis vectors form a right-handed coordinate system."""
    # Cross product of x and y
    cross_product = np.cross(x, y)

    # Check whether the cross product matches z
    return np.allclose(cross_product, z)




def caluate_rotation_matrix(coord_a, coord_b):
    flag1 = is_right_hand_system(coord_a["x"],coord_a["y"],np.array([0,0,1]))
    flag2 = is_right_hand_system(coord_b["x"],coord_b["y"],np.array([0,0,1]))
    if flag1 != flag2:
        raise ValueError("the two coordinate system is not in the same hand system")
    else:
        trans_matrix = compute_trans_matrix_a2b(coord_a,coord_b)
        result = None
        if len(coord_a["x"]) == 2:
            rotation_angle = np.arctan2(trans_matrix[1,0],trans_matrix[0,0])
            result = rotation_angle
        elif len(coord_a["x"]) == 3:
            rotation_matrix = RR.from_matrix(trans_matrix)
            result = rotation_matrix
        return result

def compute_trans_matrix_a2b(coord_a, coord_b):
    """Compute the 2×2 transformation matrix from coordinate system A to B."""
    # Build basis matrices (each column is a basis vector)
    matrix_a = np.array([coord_a["x"], coord_a["y"]]).T
    matrix_b = np.array([coord_b["x"], coord_b["y"]]).T

    # Transformation matrix
    transformation_matrix = matrix_b @ np.linalg.inv(matrix_a)

    return transformation_matrix

def compute_trans_matrix_b2a_by_a2b(T_a2b):
    T_b2a = np.linalg.inv(T_a2b)
    return T_b2a

def transform_point_a2b(point_a,T_a2b):
    if point_a.ndim == 1:
        return T_a2b @ point_a
    elif point_a.ndim == 2:
        return point_a @ T_a2b.T
    else:
        raise ValueError(f"point_a {point_a.shape} must be a 1D or 2D array, transform matrix shape is {T_a2b.shape}")


def get_euler_from_rotate_matrix(R: np.ndarray) -> List[float]:
    """Return Euler angles (radians) from a rotation matrix."""
    euler_type = "XYZ"
    sciangle = RR.from_matrix(R).as_euler(euler_type)
    return [float(angle) for angle in sciangle]


def spherical_to_cartesian(phi, theta, r, input_type="degree",origin=None):
    if input_type == "degree":
        phi = math.radians(phi)
        theta = math.radians(theta)
    x, y, z = coordinates.spherical_to_cartesian(r, phi, theta)
    x, y, z = x.value, y.value, z.value
    if origin is not None:
        x = x + origin[0]
        y = y + origin[1]
        z = z + origin[2]
    return x, y, z


# return phi theta
def cartesian_to_spherical(x, y, z, to_degree=True, return_r=False,origin=None):
    if origin is not None:
        x = x - origin[0]
        y = y - origin[1]
        z = z - origin[2]
    r, latitude, longitude = coordinates.cartesian_to_spherical(x, y, z)  #
    latitude, longitude, r = latitude.value, longitude.value, r.value
    if to_degree:
        latitude = math.degrees(latitude)
        longitude = math.degrees(longitude)
    if return_r:
        return latitude, longitude, r
    return latitude, longitude
