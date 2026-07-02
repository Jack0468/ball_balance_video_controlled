import numpy as np

# Physical constraints in mm
D_MM = 86.2  # Distance from upper limb to center of platform
E_MM = 50.0  # Length of lower limb (motor horn)
F_MM = 87.0  # Length of upper limb (connecting rod)

def calculate_triangle(theta_deg, phi_deg, h):
    """
    Calculates the target XYZ coordinates of the 3 platform joints based on desired tilt and height.
    theta: Pitch (X-axis rotation)
    phi: Roll (Y-axis rotation)
    """
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    sqrt3_half = np.sqrt(3) / 2

    # Initial vertices at height h
    v_initial = np.array([
        [0, D_MM, h],
        [-D_MM * sqrt3_half, -D_MM * 0.5, h],
        [D_MM * sqrt3_half, -D_MM * 0.5, h]
    ])

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ])

    Ry = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ])

    R = Ry @ Rx

    # The C++ legacy code only rotates the Z coordinate and maintains X/Y to avoid sliding
    v_rotated = np.copy(v_initial)
    v_rotated[:, 2] = [np.dot(R[2], v) for v in v_initial]

    # Shift Z so the centroid height remains at h
    centroid_initial_z = np.mean(v_initial[:, 2])
    centroid_rotated_z = np.mean(v_rotated[:, 2])
    z_shift = centroid_initial_z - centroid_rotated_z
    v_rotated[:, 2] += z_shift

    return v_rotated

def ik_solver(base_point, top_point):
    """
    Solves the inverse kinematics for a single leg.
    Returns the motor target angle in degrees.
    """
    r = top_point - base_point
    d2 = np.linalg.norm(r)

    # Check unreachable
    if d2 < 0.001 or d2 > E_MM + F_MM or d2 < abs(E_MM - F_MM):
        return 0.0 

    e2 = E_MM * E_MM
    f2 = F_MM * F_MM
    d2_2 = d2 * d2

    phi2 = np.arctan2(r[2], np.sqrt(r[0]**2 + r[1]**2))
    
    # Law of cosines
    val = (e2 + d2_2 - f2) / (2 * E_MM * d2)
    val = np.clip(val, -1.0, 1.0) # Prevent NaN from float precision issues
    beta = np.arccos(val)

    theta2 = phi2 - beta
    return np.degrees(theta2)

def get_target_angles(theta_deg, phi_deg, h):
    """
    Given a desired platform pitch (theta), roll (phi), and height (h),
    returns the required angles for the 3 stepper motors (A, B, C)
    """
    sqrt3_half = np.sqrt(3) / 2
    
    # Base motor joints at z=0
    base_points = np.array([
        [0, D_MM, 0],
        [-D_MM * sqrt3_half, -D_MM * 0.5, 0],
        [D_MM * sqrt3_half, -D_MM * 0.5, 0]
    ])

    top_points = calculate_triangle(theta_deg, phi_deg, h)

    theta_a = ik_solver(base_points[0], top_points[0])
    theta_b = ik_solver(base_points[1], top_points[1])
    theta_c = ik_solver(base_points[2], top_points[2])

    return theta_a, theta_b, theta_c
