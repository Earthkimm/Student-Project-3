import numpy as np

mp  = 0.25     #kg      Mass of pedulum
mc  = 6.28     #kg      Mass of cart
g   = 9.82     #m/s^2   Gravitational acceleration
l   = 0.282    #m       Length of rod
a1  = 0.0005   #        Viscous friction coeffitient
Ffc = 3.2      #        Coulomb force on cart
Ffp = 0.0041   #        Coulomb force on pendulum
n   = 4        #        number of states

A = np.array([[0, 0, 1, 0],                     # Define the A-matrix
              [0, 0, 0, 1], 
              [0, (-mp*g)/mc, (Ffc)/mc, (a1+Ffp)/(mc*l)], 
              [0, g*(mc-mp)/(mc*l), Ffc/(mc*l), (a1+Ffp)*(mc+mp)/(mp*mc*l*l)]])

B = np.array([[0], [0], [-1/mc], [-1/(l*mc)]])  # Define the B-vector

C = np.array([[1, 0, 0, 0],                     # Define the C-matrix
              [0, 1, 0, 0]])

def cont(A, B, n):
    """
    This function calculates the controllability-matrix for the system
    """
    cont = B                                        # First column of the controllability-matrix
    for i in range(n-1):                            # For the rest of the states
        M = np.linalg.matrix_power(A, i+1).dot(B)        # Calculate the next column
        cont = np.concatenate((cont, M), axis=1)    # Input the calculated column to the controllability-matrix
    return cont, np.linalg.matrix_rank(cont)             # Return the controllability-matrix and it's rank

def obs(A, C, n):
    """
    This function calculates the observability-matrix for the system
    """
    obs = C                                         # First submatrix of the observability-matrix
    for i in range(n-1):                            # For the rest of the states
        M = C.dot(np.linalg.matrix_power(A, i+1))        # Calculate the next submatrix
        obs = np.concatenate((obs, M), axis = 0)    # Input the calculated submatrix to the observability-matrix
    return obs, np.linalg.matrix_rank(obs)               # Return the observability-matrix and it's rank

cont_matrix = cont(A, B, n)     # Calculate the controllability-matrix
obs_matrix = obs(A, C, n)       # Calculate the observability-matrix

print(f'Controllabillity matrix:\n {cont_matrix[0]} \n Rank of controllability matrix: {cont_matrix[1]}\n')
print(f'Observability matrix:\n {obs_matrix[0]} \n Rank of observability matrix: {obs_matrix[1]}\n')
