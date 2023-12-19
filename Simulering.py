"""
This module plots a single simulation either with or without data.
Change Q and R to match the data from the filepath in DP.plot(<filepath>).
If a simulation without experimental data is wished, change 'use_data' to 'False' (line 70).
If 'use_data' is set to 'True' remember to change the filepath in line 74 to the wished data.
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

def RK4(f, K, t0, z_0, t_final, N):
    """
    This function performs RK4 on the function 'f' with initial values in 'z_0',
    from 't0' to 't_final' with 'N' steps.
    """
    h = (t_final - t0)/N                        # Define stepsize
    t = [t0]; z = [z_0]                          # Define lists of time and states at given time with inital values stored
    tn = t0; zn = z_0                            # Define stepvariables with initial values
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
    u=-K.dot(z)[0,0]    # Calculate control-law from previous output
    z1dot = z[2,0]
    z2dot = z[3,0]
    z3dot = (m_p*g*np.sin(z[1,0])*np.cos(z[1,0])+((alpha*z[3,0]+np.tanh(z[3,0])*F_f2)/l)*np.cos(z[1,0])-m_p*l*z[3,0]**2*np.sin(z[1,0])+ np.tanh(z[2,0])*F_f1-u)/(m_c+m_p-m_p*np.cos(z[1,0])**2)
    z4dot = (g*np.sin(z[1,0]))/l + (alpha*z[3,0]+np.tanh(z[3,0])*F_f2)/m_p*l*l + (np.cos(z[1,0])/l)*(m_p*g*np.sin(z[1,0])*np.cos(z[1,0])+((alpha*z[3,0]+np.tanh(z[3,0])*F_f2)/l)*np.cos(z[1,0])-m_p*l*z[3,0]**2*np.sin(z[1,0])+np.tanh(z[2,0])*F_f1-u)/(m_c+m_p-m_p*np.cos(z[1,0])**2)
    return np.array([[z1dot], [z2dot], [z3dot], [z4dot]])

###################################################
#
#               DEFINE PARAMETERS
#
###################################################

# Defining constants for the system
m_p = 0.25      #kg     Pendulum mass
m_c = 6.28      #kg     Cart mass
g = 9.82        #m/s^2  Gravitational acceleration
l = 0.282       #m      Pendulum rod length
K_m = 0.0934    #Nm/A   Torque constant
r_pr = 0.028    #m      Pulley radius
alpha = 0.0005  #       Pendulum drag coeffecient
F_f1 = 3.2      #       Cart Couloumb force
F_f2 = 0.0041   #       Pendulum Couloumb force

# Initial states
z_0 = np.array([[0],        # x_0
               [np.pi/8],   # theta_0 (Change to 0.15 to match data)
               [0],         # xdot_0
               [0]])        # thetadot_0

use_data = True             # Whether to use data or not

# Define time values need to perform RK4 and plotting
if use_data:        # Create time values from data
    data_t, axs = DP.plot("data/R0.01.txt")  # Save time data and get plots
    t0 = data_t[0]          # s
    t_final = data_t[-1]    # s
else:               # If not from data, create custom time values and a new figure
    fig, axs = plt.subplots(2, 2, figsize=(16/2, 9/2))  # Create new subplots in a 2 by 2 grid
    t0 = 0                  # s
    t_final = 3             # s
N = 10000       # Number of RK4 steps

###################################################
#
#               DEFINE MATRICES
#
###################################################

A = np.array([[0, 0, 1, 0],     # A-matrix
              [0, 0, 0, 1],
              [0, (-m_p*g)/m_c, (F_f1)/m_c, (alpha+F_f2)/(m_c*l)],
              [0, g*(m_c-m_p)/(m_c*l), F_f1/(m_c*l), (alpha+F_f2)*(m_c+m_p)/(m_p*m_c*l*l)]])
B = np.array([[0],              # B-vector
              [0],
              [-1/m_c],
              [-1/(l*m_c)]])
Q = np.diag([5000, 1, 1, 1])    # Q-matrix
R = np.array([0.01])            # R-scalar

###################################################
#
#               CALCULATE STATES
#
###################################################

# Perform LQR with python's control-module
K, S, E = control.lqr(A, B, Q, R)
print(f"The gain matrix is calculated as:\n{K}\n")
print(f"The eigenvalues for (A-BK) are:\n{E}\n")

t, z = RK4(f, K, t0, z_0, t_final, N)   # Perform RK4 and save results

# Calculate the current input from each state-vector in z
curr = (K.dot(z)*r_pr/K_m).flatten()    # i = K * z * r / K_m, uses .flatten() to transform the 3D array result into 1D array

print(f"The largest deviation from 0 of the cart is:\n{np.max(abs(z[:,0]))}\n") # Largest cart movement
print(f"The highest current input is:\n{np.max(abs(curr))}\n")                  # Highest current input

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

