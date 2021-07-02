# Knee Flexion Calculation 
# based on Jakob, et. al

import pandas as pd
import numpy as np
from numpy import cos, sin, tan, arccos, pi

# Part One: functional alignment procedure
""" rotation matrix that transforms sensor frame at any time step into initial sensor frame 
multiplication of single rotation matrices around each axis of U, V, and W """

# theta = rotation around V
# phi = rotation around U
# psi = rotation around W

""" R_i = np.array([[cos(theta)*sin(psi), -cos(theta)*sin(psi), sin(theta)], 
[cos(theta)*sin(psi)+sin(phi)*sin(theta)*cos(psi),cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi), -sin(phi)*cos(theta)], 
[sin(phi)*sin(psi) - cos(phi)*sin(theta)*cos(psi), sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi), cos(phi)*cos(theta)]]) """


""" vertical alignment of thigh and shank sensor frames to Z-axis
motion still standing where subject stands with straight legs for 10 seconds """

# from data, we are given average gravitational vector 'g'
# first, compute 'alpha' which is angle of misalignment between 'g' and Z axis
Z = [0,0,1]
g = [0,1,0] # assume no misalignment
mag_Z = np.linalg.norm(Z)
mag_g = np.linalg.norm(g)
alpha = arccos( float(np.dot(Z, g)) / (mag_Z*mag_g) )

# next, define 'k' which is the unit 3D vector perpendicular to 'g' and Z
k = np.cross(Z, g) / np.linalg.norm(np.cross(Z,g))
kx = k[0] # x component of k
ky = k[1] # y component of k
kz = k[2] # z component of k

# define 'v'
v = 1 - cos(alpha)
ca = cos(alpha)
sa = sin(alpha)

#finally calculate rotation matrix to align vertical components W and w with Z
R_Z = np.array([[kx*kx*v*alpha + ca, kx*ky*v*alpha - kz*sa, kx*kz*v*alpha + ky*sa],
       [kx*ky*v*alpha + kz*sa, ky*ky*v*alpha + ca, ky*kz*v*alpha - kx*sa],
       [kx*kz*v*alpha - ky*sa, ky*kz*v*alpha + kx*sa, kz*kz*v*alpha + ca]])

print(R_Z)

""" horizontal alignment 
straight leg is lifted up and down laterally for 20 seconds to produce 
constantly oriented angular rate vector in thigh and shank sensor """

# will need to change this code depending on how the data is given to us
# given angular rate vectors w1 (from thigh) and w2 (from shank) at each time step
w1 = [[0,0,1],[1,2,3],[1,1,2]] # random list of vectors
w2 = [[0,0,2],[1,2,4],[0,2,3]] # random list of vectors

# new list of misaligned angles for each time step
B_k = []
w1_x = []
w1_y = []
for i in range((len(w1)-1)):
  # calculate misalignment angle at specific time step
  mag_w1 = np.linalg.norm(w1[i])
  mag_w2 = np.linalg.norm(w2[i])
  new_Bk = arccos( float(np.dot(w1[i], w2[i])) / (mag_w1*mag_w2) )
  B_k.append(new_Bk)
  w1_x.append(w1[i][0]) # list of w1_x at each time step
  w1_y.append(w1[i][1]) # list of w1_y at each time step

# weighted average of misalignment angle
num = 0
den = 0
for k in range((len(B_k)-1)):
  num += abs(w1_x[k]*w1_y[k]) * B_k[k]
  den += np.linalg.norm(w1[k])
B = num/den

# compute rotation matrix for horizontal alignment
R_XY = np.array([[cos(B), -sin(B), 0],
        [sin(B), cos(B), 0],
        [0, 0, 1]])

# Part Two: extended Kalman filter (EKF) estimate IMU relative orientation

# define algorithm parameters 
sigma_a = 0.1
sigma_w = 0.1
r = 0.1
dt = 0.1

# create noise matrix 'R' which is 6X6 Identity matrix
R = np.array([[r, 0, 0, 0, 0, 0],
              [0, r, 0, 0, 0, 0],
              [0, 0, r, 0, 0, 0],
              [0, 0, 0, r, 0, 0],
              [0, 0, 0, 0, r, 0],
              [0, 0, 0, 0, 0, r]])

# create covariance matrix 'Q' which is 8X8 matrix
Q = np.array([[sigma_a, 0, 0, 0, 0, 0, 0, 0],
     [0, sigma_a, 0, 0, 0, 0, 0, 0],
     [0, 0, sigma_a, 0, 0, 0, 0, 0],
     [0, 0, 0, sigma_w, 0, 0, 0, 0],
     [0, 0, 0, 0, sigma_w, 0, 0, 0],
     [0, 0, 0, 0, 0, sigma_w, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0]])

# initial P matrix (covariance estimate)
shape = (8, 8) # shape of covariance estimate
P_k0 = np.zeros(shape) # something by something zero matrix

# create function of the dynamic system 'f_propogate'
def f_propogate(au_k0, av_k0, aw_k0, wu_k0, wv_k0, ww_k0, phi_k0, theta_k0, u1, 
                u2, u3, u4, u5, u6):
  """ a_k0 is previous accelermeter data
  w_k0 is previous gyroscope data
  phi_k0 is previous phi calculation or initial prediction
  theta_k0 is previous theta calculation or initial prediction
  u (1-3) is previous accelerometer noise
  u (4-6) is previous gyroscope noise 
  takes in previous data and noise data to predict data at next time step"""
  # below defines all the new propogated components
  au = au_k0 + u1 
  av = av_k0 + u2
  aw = aw_k0 + u3
  wu = wu_k0 + u4
  wv = wv_k0 + u5
  ww = ww_k0 + u6
  phi = phi_k0 + dt*(wu_k0 + tan(theta_k0)*wv_k0*sin(phi_k0))
  theta = theta_k0 + dt*(wv_k0*cos(phi_k0))
  # propogated state vector
  f = np.array([[au],
                [av],
                [aw],
                [wu],
                [wv],
                [ww],
                [phi],
                [theta]])
  return f

# calculate Jacobian matrices 'F' and 'H'
def F_jacobian(phi, theta, wv):
  row7_col5 = dt*sin(phi)*tan(theta)
  row8_col5 = dt*cos(phi)
  row7_col7 = 1 + cos(phi)*dt*tan(theta)*wv
  row8_col7 = -dt*wv*sin(phi)
  row7_col8 = (dt*tan(theta)*wv)/(cos(theta))
  F_jacobian = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, dt, row7_col5, 0, row7_col7, row7_col8],
                         [0, 0, 0, 0, row8_col5, 0, row8_col7, 1]])
  return F_jacobian
  
# not sure if H jacobian is correct...
H_jacobian = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0]])

# EKF for thigh
# initialize algorithm with data
tau_k0 = [0, 1, 2, 2, 4, 7] # random data
tav_k0 = [0, 1, 0, 0, 0, 0] # random data
taw_k0 = [0, 3, 2, 0, 0, 1] # random data
twu_k0 = [0, 0, 0, 0, 0, 0] # random data
twv_k0 = [0, 1, 2, 0, 1, 3] # random data
tww_k0 = [0, 0, 0, 0, 0, 0] # random data
tphi_k0 = pi
ttheta_k0 = pi

tf_k0 = np.array([[tau_k0[0]],
               [tav_k0[0]],
               [taw_k0[0]],
               [twu_k0[0]],
               [twv_k0[0]],
               [tww_k0[0]],
               [tphi_k0],
               [ttheta_k0]])

phi1 = []
theta1 = []

for k in range(1, len(tau_k0)):
  # get new predicted measurement (8x8 matrix)
  fk = f_propogate(tf_k0[0][0], tf_k0[1][0], tf_k0[2][0], tf_k0[3][0], 
                   tf_k0[4][0], tf_k0[5][0], tf_k0[6][0],
                   tf_k0[7][0], 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
  # compute F jacobian
  F = F_jacobian(tf_k0[6][0], tf_k0[7][0], tf_k0[4][0]) # 8x8 matrix
  # get new predicted covariance estimate
  P = (F @ P_k0 @ np.transpose(F)) + Q # 8x8 matrix
  # update measurement 
  th_k = np.array([[tau_k0[k]],
               [tav_k0[k]],
               [taw_k0[k]],
               [twu_k0[k]],
               [twv_k0[k]],
               [tww_k0[k]]])
  tv_k = np.array([[0.1],
               [0.1],
               [0.1],
               [0.1],
               [0.1],
               [0.1]]) # measurement noise
  tz_k = th_k + tv_k
  # update covariance
  S = (H_jacobian @ P @ np.transpose(H_jacobian)) + R # 6x6 matrix
  # calculate Kalman gain 'K'
  K = P @ np.transpose(H_jacobian) @ np.linalg.inv(S) 
  # new state estimate
  tf_k0 = tf_k0 + (K @ tz_k)
  # new covariance estimate
  P = (np.identity(8) - (K @ H_jacobian) @ P)
  # update angle data to collect
  phi1.append(tf_k0[6][0])
  theta1.append(tf_k0[7][0])

# EKF for shank

# initialize algorithm with data
sau_k0 = [0.5, 1, 2, 2, 4, 7] # random data
sav_k0 = [2, 1, 0, 0, 0, 0] # random data
saw_k0 = [0, 3, 2, 0, 0, 1] # random data
swu_k0 = [0.5, 0, 0, 0, 0, 0] # random data
swv_k0 = [0, 1, 2, 0, 1, 3] # random data
sww_k0 = [0, 0, 0, 0, 0, 0] # random data
sphi_k0 = pi
stheta_k0 = pi

sf_k0 = np.array([[sau_k0[0]],
               [sav_k0[0]],
               [saw_k0[0]],
               [swu_k0[0]],
               [swv_k0[0]],
               [sww_k0[0]],
               [sphi_k0],
               [stheta_k0]])

# get angles
phi2 = []
theta2 = []

for k in range(1, len(sau_k0)):
  # get new predicted measurement (8x8 matrix)
  fk = f_propogate(sf_k0[0][0], sf_k0[1][0], sf_k0[2][0], sf_k0[3][0], 
                   sf_k0[4][0], sf_k0[5][0], sf_k0[6][0],
                   sf_k0[7][0], 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
  # compute F jacobian
  F = F_jacobian(sf_k0[6][0], sf_k0[7][0], sf_k0[4][0]) # 8x8 matrix
  # get new predicted covariance estimate
  P = (F @ P_k0 @ np.transpose(F)) + Q # 8x8 matrix
  # update measurement 
  sh_k = np.array([[sau_k0[k]],
               [sav_k0[k]],
               [saw_k0[k]],
               [swu_k0[k]],
               [swv_k0[k]],
               [sww_k0[k]]])
  sv_k = np.array([[0.1],
               [0.1],
               [0.1],
               [0.1],
               [0.1],
               [0.1]]) # measurement noise
  sz_k = sh_k + sv_k
  # update covariance
  S = (H_jacobian @ P @ np.transpose(H_jacobian)) + R # 6x6 matrix
  # calculate Kalman gain 'K'
  K = P @ np.transpose(H_jacobian) @ np.linalg.inv(S) 
  # new state estimate
  sf_k0 = sf_k0 + (K @ sz_k)
  # new covariance estimate
  P = (np.identity(8) - (K @ H_jacobian) @ P)
  # update angle data to collect
  phi2.append(sf_k0[6][0])
  theta2.append(sf_k0[7][0])


# Part Three: flexion/ extension knee angle calculation
knee_angle = []

for k in range(0, len(phi1)):
  # vertical component of sensor frame for the thigh
  ri1 = np.array([[sin(theta1[k])],
         [-sin(phi1[k])*cos(theta1[k])], 
         [cos(phi1[k])*cos(theta1[k])]])
  # vertical component of sensor frame for shank
  ri2 = np.array([[sin(theta1[k])],
                       [-sin(phi2[k])*cos(theta2[k])], 
                       [cos(phi2[k])*cos(theta2[k])]])
  
  # angles in the JCS
  R_Z1 = R_Z # rotation matrices should be unique for the thigh and shank but I'm assuming the same 
  R_Z2 = R_Z
  r1k = R_Z1 @ ri1
  r2k = R_XY @ R_Z2 @ ri2

  # only need the y and z components to vector to do calculation for knee angle
  r1 = [r1k[1][0], r1k[2][0]]
  r2 = [r2k[1][0], r2k[2][0]]
  mag_r1 = np.linalg.norm(r1)
  mag_r2 = np.linalg.norm(r2)
  arccos_calc = arccos( float(np.dot(r1, r2)) / (mag_r1*mag_r2) )
  sign = np.sign(r2[0]*r1[1] - r2[1]*r1[0])
  angle = sign*arccos_calc
  knee_angle.append(angle)

print(knee_angle)

