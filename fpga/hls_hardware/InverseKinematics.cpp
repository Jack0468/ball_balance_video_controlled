#include "InverseKinematics.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

//constants in mm. d = distance from upper limb to center of platform, e = length of lower limb, f = length of upper limb
const fixed_t d = 86.2;
const fixed_t e = 50.0;
const fixed_t f = 87.0;

// Helper function to copy 3D vector
void copy_vector(fixed_t dest[3], const fixed_t src[3]) {
#pragma HLS PIPELINE
  for (int i = 0; i < 3; i++) {
    dest[i] = src[i];
  }
}

// Helper function to add two 3D vectors
void add_vectors(fixed_t result[3], const fixed_t a[3], const fixed_t b[3]) {
#pragma HLS PIPELINE
  for (int i = 0; i < 3; i++) {
    result[i] = a[i] + b[i];
  }
}

// Helper function to subtract two 3D vectors
void subtract_vectors(fixed_t result[3], const fixed_t a[3], const fixed_t b[3]) {
#pragma HLS PIPELINE
  for (int i = 0; i < 3; i++) {
    result[i] = a[i] - b[i];
  }
}

// Helper function to calculate vector magnitude
fixed_t vector_magnitude(const fixed_t v[3]) {
  return hls::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// General 3x3 matrix multiplication function
void matrix_multiply_3x3(const fixed_t A[3][3], const fixed_t B[3][3], fixed_t result[3][3]) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE
      result[i][j] = 0;
      for (int k = 0; k < 3; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// Matrix-vector multiplication (3x3 matrix * 3x1 vector = 3x1 vector)
void matrix_vector_multiply_3x3(const fixed_t matrix[3][3], const fixed_t vector[3], fixed_t result[3]) {
  for (int i = 0; i < 3; i++) {
#pragma HLS PIPELINE
    result[i] = 0;
    for (int j = 0; j < 3; j++) {
      result[i] += matrix[i][j] * vector[j];
    }
  }
}

// Single row of matrix multiplied by vector (returns scalar)
fixed_t matrix_row_vector_multiply(const fixed_t matrix_row[3], const fixed_t vector[3]) {
#pragma HLS PIPELINE
  fixed_t result = 0;
  for (int i = 0; i < 3; i++) {
    result += matrix_row[i] * vector[i];
  }
  return result;
}

// Function to calculate rotated triangle.
Triangle calculate_triangle(fixed_t theta_deg, fixed_t phi_deg, fixed_t d, fixed_t h) {

  // Convert to radians
  fixed_t theta = theta_deg * (fixed_t)(PI / 180.0);
  fixed_t phi = phi_deg * (fixed_t)(PI / 180.0);

  // Precompute trigonometric values
  fixed_t cos_theta = hls::cos(theta), sin_theta = hls::sin(theta);
  fixed_t cos_phi = hls::cos(phi), sin_phi = hls::sin(phi);
  fixed_t sqrt3_half = hls::sqrt((fixed_t)3) / (fixed_t)2;

  // Create initial triangle vertices
  fixed_t vertices_i[3][3] = {
    { 0, d, h },                       // VA_i
    { -d * sqrt3_half, -d * 0.5, h },  // VB_i
    { d * sqrt3_half, -d * 0.5, h }    // VC_i
  };

  // Calculate centroid of initial triangle
  fixed_t centroid_target[3] = {
    (vertices_i[0][0] + vertices_i[1][0] + vertices_i[2][0]) / (fixed_t)3.0,
    (vertices_i[0][1] + vertices_i[1][1] + vertices_i[2][1]) / (fixed_t)3.0,
    (vertices_i[0][2] + vertices_i[1][2] + vertices_i[2][2]) / (fixed_t)3.0
  };

  // Rotation matrices around x and y axes
  fixed_t Rx[3][3] = {
    { 1, 0, 0 },
    { 0, cos_theta, -sin_theta },
    { 0, sin_theta, cos_theta }
  };

  fixed_t Ry[3][3] = {
    { cos_phi, 0, sin_phi },
    { 0, 1, 0 },
    { -sin_phi, 0, cos_phi }
  };

  // Combined rotation matrix R = Ry * Rx
  fixed_t R[3][3];
  matrix_multiply_3x3(Ry, Rx, R);

  // Create result triangle with rotated z-coordinates
  Triangle result;
  fixed_t* result_vertices[3] = { result.VA, result.VB, result.VC };

  for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
    result_vertices[i][0] = vertices_i[i][0];  // Keep x unchanged
    result_vertices[i][1] = vertices_i[i][1];  // Keep y unchanged
    // Use the third row of rotation matrix to compute rotated z
    result_vertices[i][2] = matrix_row_vector_multiply(R[2], vertices_i[i]);
  }

  // Calculate centroid of rotated triangle
  fixed_t centroid_f[3] = {
    (result.VA[0] + result.VB[0] + result.VC[0]) / (fixed_t)3.0,
    (result.VA[1] + result.VB[1] + result.VC[1]) / (fixed_t)3.0,
    (result.VA[2] + result.VB[2] + result.VC[2]) / (fixed_t)3.0
  };

  // Calculate translation vector
  fixed_t translate[3];
  subtract_vectors(translate, centroid_target, centroid_f);

  // Apply translation to all vertices
  for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
    add_vectors(result_vertices[i], result_vertices[i], translate);
  }

  return result;
}

// IK solver function
IKResult ik_solver(fixed_t base_point[3], fixed_t top_point[3], fixed_t e, fixed_t f) {

  // Calculate r vector
  fixed_t r[3];
  subtract_vectors(r, top_point, base_point);

  // Calculate distance
  fixed_t d2 = vector_magnitude(r);

  // Check if target is unreachable
  fixed_t diff = e - f;
  if (diff < 0) diff = -diff; // absolute value
  
  if (d2 < (fixed_t)0.001 || d2 > e + f || d2 < diff) {
    IKResult err = {0, 0};
    return err;  // Error case
  }

  // Calculate angles
  fixed_t e2 = e * e, f2 = f * f, d2_2 = d2 * d2;
  fixed_t alpha = hls::acos((e2 + f2 - d2_2) / (2 * e * f));
  fixed_t phi2 = hls::atan2(r[2], hls::sqrt(r[0] * r[0] + r[1] * r[1]));
  fixed_t beta = hls::acos((e2 + d2_2 - f2) / (2 * e * d2));
  fixed_t theta2 = phi2 - beta;

  IKResult res = { alpha * (fixed_t)(180.0 / PI), theta2 * (fixed_t)(180.0 / PI) };
  return res;
}

//gets angles, using ik_solver function and calculate_triangle function. theta will balance y axis, phi will balance x axis
CalculatedAngles get_angles(fixed_t theta, fixed_t phi, fixed_t h) {

  // Base points - initial triangle vertices with z = 0
  fixed_t sqrt3_half = hls::sqrt((fixed_t)3) / (fixed_t)2;
  fixed_t base_VA[3] = { 0, d, 0 };                       // VA_i with z = 0
  fixed_t base_VB[3] = { -d * sqrt3_half, -d * 0.5, 0 };  // VB_i with z = 0
  fixed_t base_VC[3] = { d * sqrt3_half, -d * 0.5, 0 };   // VC_i with z = 0

  //calculate rotated triangle from the given inputs
  Triangle tri;
  tri = calculate_triangle(theta, phi, d, h);

  // Top points - results from calculate_triangle
  fixed_t top_VA[3] = { tri.VA[0], tri.VA[1], tri.VA[2] };
  fixed_t top_VB[3] = { tri.VB[0], tri.VB[1], tri.VB[2] };
  fixed_t top_VC[3] = { tri.VC[0], tri.VC[1], tri.VC[2] };

  // Call ik_solver for each vertex pair and get theta2 values
  IKResult angles_A = ik_solver(base_VA, top_VA, e, f);
  IKResult angles_B = ik_solver(base_VB, top_VB, e, f);
  IKResult angles_C = ik_solver(base_VC, top_VC, e, f);

  //create result
  CalculatedAngles result;
  result.thetaA = angles_A.theta2;
  result.thetaB = angles_B.theta2;
  result.thetaC = angles_C.theta2;

  return result;
}
