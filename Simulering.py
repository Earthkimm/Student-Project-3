"""
This module plots a single simulation either with or without data.
Change Q and R to match the data from the filepath in DP.plot(<filepath>).
If no data is wished change 'use_data' to 'False'.
"""
import numpy as np
import matplotlib.pyplot as plt
import control
import Data_plotting as DP

###################################################
#
#               DEFINE FUNCTIONS
#
###################################################

def RK4(f, K, t0, z0, t_final, N):
    """
    This function performs RK4 on the function 'f' with initial values in 'z0',
    from 't0' to 't_final' with 'N' steps.
    """
    h = (t_final - t0)/N                        # Define stepsize
    t = [t0]; z = [z0]                          # Define lists of time and states at given time with inital values stored
    tn = t0; zn = z0                            # Define stepvariables with initial values
    for k in range(N):
        k1 = f(zn, K)                           # The following four lines calculates k1, k2, k3 and k4 used for RK4
        k2 = f(zn + h*k1/2, K)
        k3 = f(zn + h*k2/2, K)
        k4 = f(zn + h*k3, K)
        tn = tn + h                             # Updates t for next step
        zn = zn + h*(k1 + 2*(k2 + k3) + k4)/6   # Updates the next statevector by evaluating RK4
        t.append(tn); z.append(zn);             # Appends the current time and statevector to their lists
    return np.array(t), np.array(z)             # Returns the lists of time and their corresponding statevectors

def f(z, K):
    """
    This function calculates zdot from the given statevector 'z' and the given gain matrix 'K'
    """
    u=-float(K.dot(z))  # Calculate control-law from previous output
    z1dot = z[2,0]
    z2dot = z[3,0]
    z3dot = (mp*g*np.sin(z[1,0])*np.cos(z[1,0])+((a1*z[3,0]+np.tanh(z[3,0])*Ffp)/l)*np.cos(z[1,0])-mp*l*z[3,0]**2*np.sin(z[1,0])+ np.tanh(z[2,0])*Ffc-u)/(mc+mp-mp*np.cos(z[1,0])**2)
    z4dot = (g*np.sin(z[1,0]))/l + (a1*z[3,0]+np.tanh(z[3,0])*Ffp)/mp*l*l + (np.cos(z[1,0])/l)*(mp*g*np.sin(z[1,0])*np.cos(z[1,0])+((a1*z[3,0]+np.tanh(z[3,0])*Ffp)/l)*np.cos(z[1,0])-mp*l*z[3,0]**2*np.sin(z[1,0])+np.tanh(z[2,0])*Ffc-u)/(mc+mp-mp*np.cos(z[1,0])**2)
    return np.array([[z1dot], [z2dot], [z3dot], [z4dot]])

###################################################
#
#               DEFINE PARAMETERS
#
###################################################

# Defining constants for the system
mp = 0.25   #kg     Mass of pendulum
mc = 6.28   #kg     Mass of cart
g = 9.82    #m/s^2  Gravitational acceleration
l = 0.282   #m      Length of rod
km = 0.0934 #Nm/A   Motorconstant
r = 0.028   #m      Radius of pulley-gear
a1 = 0.0005 #       Viscous friction coeffecient
Ffc = 3.2   #       Couloumb friction coeffecient
Ffp = 0.0041#       F_fp

# Initial states
z0 = np.array([[0],         # x_0
               [0.15],      # theta_0
               [0],         # xdot_0
               [0]])        # thetadot_0

use_data = True             # Whether to use data or not
if use_data:
    data_t, axs = DP.plot("data\Bryson20nr2.txt")  # Save time data and get plots

# Define time values need to perform RK4 and plotting
if use_data:        # Create time values from data
    t0 = data_t[0]          # s
    t_final = data_t[-1]    # s
else:               # If not from data, create custom time values and a new figure
    t0 = 0                  # s
    t_final = 3             # s
    fig, axs = plt.subplots(2, 2, figsize=(16/2, 9/2))  # Create new subplots in a 2 by 2 grid
N = 10000       # Number of steps

###################################################
#
#               DEFINE MATRICES
#
###################################################

A = np.array([[0, 0, 1, 0],     # A-matrix
              [0, 0, 0, 1],
              [0, (-mp*g)/mc, (Ffc)/mc, (a1+Ffp)/(mc*l)],
              [0, g*(mc-mp)/(mc*l), Ffc/(mc*l), (a1+Ffp)*(mc+mp)/(mp*mc*l*l)]])
B = np.array([[0],              # B-vector
              [0],
              [-1/mc],
              [-1/(l*mc)]])
Q = np.diag([5000, 1, 1, 1])    # Q-matrix
R = np.array([0.005])            # R-scalar

###################################################
#
#               CALCULATE STATES
#
###################################################

# Perform LQR with python's control-module
K, S, E = control.lqr(A, B, Q, R)
print(K)
print(E)

t, z = RK4(f, K, t0, z0, t_final, N)   # Perform RK4 and save results

# Calculate the current input from each state-vector in z
curr = (K.dot(z)*r/km).flatten()    # i = K * z * r / km, uses .flatten() to transform the 3D array result into 1D array

print(np.max(abs(z[:,0])))  # Largest cart movement
print(np.max(abs(curr)))    # Highest current input

###################################################
#
#                   PLOTTING
#
###################################################

# Create variables and loop for plotting states
index = 0   # Used for accesing correlated state-vector, label and title
ylabels = ['$x$', '$\\theta$', '$\dot{x}$', '$\dot{\\theta}$']              # Label on y-axis of each plot
titles = ['Cart Position', 'Angle', 'Cart Velocity', 'Angular Velocity']    # Title for each plot
for row in axs:
    for subplot in row:
        subplot.plot(t, z[:,index], label="Simulation", c="tab:blue")   # Plot state
        subplot.set_title(titles[index])                                # Give plot it's respective title
        subplot.set_ylabel(ylabels[index], rotation=0)                  # Label with respective label
        subplot.set_xlabel("t")                                         # Label x-axis
        subplot.axhline(y=0, c="gray", linestyle="--", zorder=-1, linewidth=0.9)   # Reference for y=0
        if not use_data:
            y_max = np.round(np.max(abs(z[:,index])), 1)                    # Find largest deviation from 0 rounded for prettier numbers
            subplot.set_yticks(np.linspace(-y_max, y_max, 5))               # Use largest deviation to create equidistant y-ticks
            subplot.set_ylim(-y_max*1.1, y_max*1.1)                         # Limit y-axis to show the biggest and smallest values of each plot
        subplot.set_xticks(np.linspace(t0, t_final, 6))                 # Make x-axis be 6 equidistant points
        index += 1  # Next state
# Final touches of simulated plots
if axs[0,0].get_ylim()[0] < -0.41:   # If limits go beyond physical boundaries, show them
        axs[0,0].axhline(y=-0.4, c="black", linestyle="--", linewidth=0.9)   # Vertical lines to show physical limitations of the cart
        axs[0,0].axhline(y=0.4, c="black", linestyle="--", linewidth=0.9)

# Make plots fit together and show them
plt.tight_layout()
plt.savefig('grafer.jpg', dpi = 750)
plt.show()

# Plot current input in separate plot
plt.plot(t, curr)
plt.xlabel('time, [s]')
plt.ylabel('current, [A]')
plt.show()

