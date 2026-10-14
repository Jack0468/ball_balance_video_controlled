#ifndef INVERSEKINEMATICS_H
#define INVERSEKINEMATICS_H

#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32, 16> fixed_t;

// Structure to hold triangle points
struct Triangle {
  fixed_t VA[3];
  fixed_t VB[3];
  fixed_t VC[3];
};

// Structure to hold IK solver results
struct IKResult {
  fixed_t alpha;   // elbow angle in degrees
  fixed_t theta2;  // shoulder angle in degrees
  // Set true when ik_solver's target was unreachable (d2 < 0.001 || d2 > e+f
  // || d2 < |e-f|) -- previously this case silently returned {0,0} with no
  // way for a caller to tell "leg unreachable, commanded to 0 deg" apart from
  // "leg genuinely computed 0 deg". Real fault signal, flagged in
  // docs/TODO.md (item 600) as missing; added 2026-10-14.
  bool fault;
};

//Structure to hold calculated angles for each motor. Combines calculate_triangle and ik_solver functions
struct CalculatedAngles {
  fixed_t thetaA;
  fixed_t thetaB;
  fixed_t thetaC;
  // OR of all three legs' IKResult::fault -- true if ANY leg's target was
  // unreachable this call. Does not identify which leg; a caller needing
  // per-leg fault detail should read the three IKResult values directly
  // instead of going through get_angles().
  bool fault;
};

// Function declarations
Triangle calculate_triangle(fixed_t theta_deg, fixed_t phi_deg, fixed_t d, fixed_t h);
IKResult ik_solver(fixed_t base_point[3], fixed_t top_point[3], fixed_t e, fixed_t f);
CalculatedAngles get_angles(fixed_t theta, fixed_t phi, fixed_t h);

// Matrix multiplication helper functions
void matrix_multiply_3x3(const fixed_t A[3][3], const fixed_t B[3][3], fixed_t result[3][3]);
void matrix_vector_multiply_3x3(const fixed_t matrix[3][3], const fixed_t vector[3], fixed_t result[3]);
fixed_t matrix_row_vector_multiply(const fixed_t matrix_row[3], const fixed_t vector[3]);

#endif
