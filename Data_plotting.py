"""
This module is to be called from 'Simulering.py' to create plots of data.
If a plot of only data is wanted, there are 2 outcommented lines at the bottom of the script for this.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def midling(array, k):          # k'th order averaging filter function. Takes an array and a k as input, and returns an array with the averaged values
    l = len(array)            
    midlet = np.zeros(l)        # Defines a zero-arrray of same length as inputarray
    for i in range((k-1), l):   # For loop running through k to l 
        x = 0                   # Countvariable is defined
        for j in range(k):      # The i'th and the k-1 previous entries in the inputarray is summed
            x += array[i-j]
        midlet[i] = x/k         # and is divided with k. The i'th averaged entry is now assigned to the i'th element in "midlet"
    for i in range((k-1)):      # To avoid the first k-1 entries being 0, they are set to an average of the first i entries, for each i in range(k-1)
        midlet[i]=np.sum(array[:i+1])/(i+1)
    return midlet               # Returns the averaged values

def plot(filepath):
    SAMPLING_TIME = 0.00667 # s
    # Load data from 'data/<filename>' into the data variable
    data = pd.read_csv(filepath).iloc[:,0:2][0:2900] # Limit for 'Bryson20nr2.txt' data at the point where the pendulum is stabilized
    data_points = len(data.iloc[:,0])       # Saves number of data entries
    data_cart_pos = []                      # List for cart position from the data
    data_angle = []                         # List for angle of pendulum from the data
    for i in range(len(data.iloc[:,0])):    # For each data entry do the following
        point = data.iloc[:,0][i]           # Saves the i'th cart position in a local variable
        point2 = data.iloc[:,1][i]          # Saves the i'th agnle in a local variable
        if "S" in str(point):   # If the data failed it contains the sentence 'Safety angle exceeded - Turning off controllers', skip this data entry
            data_points -= 1
            continue
        data_cart_pos.append(float(point))  # Save the cart position in the list
        data_angle.append(float(point2))    # Save the angle in the list

    data_cart_pos = midling(np.array(data_cart_pos) - data_cart_pos[0], 10) # Average out the cart position
    data_angle = midling(np.array(data_angle), 10)                          # Average out the angle
    data_cart_vel = np.zeros_like(data_cart_pos)    # Create zero-array for storing cart velocity
    data_angle_vel = np.zeros_like(data_angle)      # Create zero-array for storing angular velocity
    for i in range(1, len(data_cart_pos)):  # For every data point do this:
        data_cart_vel[i] = (data_cart_pos[i] - data_cart_pos[i-1]) / SAMPLING_TIME  # Calculate cart velocity from the previous sample to the current
        data_angle_vel[i] = (data_angle[i] - data_angle[i-1]) / SAMPLING_TIME       # Calculate angular velocity from the previous sample to the current
    data_states = np.array([data_cart_pos,      # Create a 2D array with each row containing info from the data
                            data_angle,
                            data_cart_vel,
                            data_angle_vel])
    data_t = np.arange(data_points)     # Makes an array of the number of samples
    data_t = data_t * SAMPLING_TIME     # Convert sample times into time in seconds

    fig, axs = plt.subplots(2, 2)           # 4 new plots for the data

    ylabels = ['$x$', '$\\theta$', '$\dot{x}$', '$\dot{\\theta}$']              # Label on y-axis of each plot
    titles = ['Cart Position', 'Angle', 'Cart Velocity', 'Angular Velocity']    # Title for each plot
    index = 0                               # Used for accesing correlated state-vector, label and title
    for row in axs:
        for subplot in row:
            subplot.plot(data_t, data_states[index], label="Data", c="tab:orange")  # Plot the current data values
            subplot.set_title(titles[index])                                # Give plot it's respective title
            subplot.set_ylabel(ylabels[index], rotation=0)                  # Label with respective label
            subplot.set_xlabel("t")                                         # Label x-axis
            subplot.axhline(y=0, c="gray", linestyle="--", zorder=-1, linewidth=0.9)   # Reference for y=0
            y_max = np.round(np.max(abs(data_states[index])), 1)            # Find largest deviation from 0 rounded for prettier numbers
            if subplot.get_yticks()[0] > -y_max:
                subplot.set_yticks(np.linspace(-y_max, y_max, 5))           # Use largest deviation to create equidistant y-ticks
                subplot.set_ylim(-y_max*1.1, y_max*1.1)                     # Limit y-axis to show the biggest and smallest values of each plot if bigger than simulation limits
            subplot.set_xticks(np.round(np.linspace(data_t[0], data_t[-1], 6), 1))    # Make x-axis be 6 equidistant points
            subplot.legend()    # Make the legend box show up in every plot
            index += 1
    # Final touches of data plots
    if axs[0,0].get_ylim()[0] < -0.41:   # If limits go beyond physical boundaries, show them
        axs[0,0].axhline(y=-0.4, c="black", linestyle="--", linewidth=0.9)   # Vertical lines to show physical limitations of the cart
        axs[0,0].axhline(y=0.4, c="black", linestyle="--", linewidth=0.9)
    plt.tight_layout()
    #plt.show()
    return data_t, axs
#plot(<filepath>)