import numpy as np

m_p = 0.25      #kg     Pendulum mass
m_c = 6.28      #kg     Cart mass
g = 9.82        #m/s^2  Gravitational acceleration
l = 0.282       #m      Pendulum rod length
alpha = 0.0005  #       Pendulum drag coeffecient
F_f1 = 3.2      #       Cart Couloumb force
F_f2 = 0.0041   #       Pendulum Couloumb force
n   = 4         #       Number of states

A = np.array([[0, 0, 1, 0],     # Define the A-matrix
              [0, 0, 0, 1], 
              [0, (-m_p*g)/m_c, (F_f1)/m_c, (alpha+F_f2)/(m_c*l)], 
              [0, g*(m_c-m_p)/(m_c*l), F_f1/(m_c*l), (alpha+F_f2)*(m_c+m_p)/(m_p*m_c*l*l)]])

B = np.array([[0],              # Define the B-vector
              [0],
              [-1/m_c],
              [-1/(l*m_c)]])

C = np.array([[1, 0, 0, 0],     # Define the C-matrix
              [0, 1, 0, 0]])

def cont(A, B, n):
    """
    This function calculates the controllability-matrix for the system
    """
    cont = B                                        # First column of the controllability-matrix
    for i in range(n-1):                            # For the rest of the states
        M = np.linalg.matrix_power(A, i+1).dot(B)   # Calculate the next column
        cont = np.concatenate((cont, M), axis=1)    # Input the calculated column to the controllability-matrix
    return cont, np.linalg.matrix_rank(cont)        # Return the controllability-matrix and it's rank

def obs(A, C, n):
    """
    This function calculates the observability-matrix for the system
    """
    obs = C                                         # First submatrix of the observability-matrix
    for i in range(n-1):                            # For the rest of the states
        M = C.dot(np.linalg.matrix_power(A, i+1))   # Calculate the next submatrix
        obs = np.concatenate((obs, M), axis = 0)    # Input the calculated submatrix to the observability-matrix
    return obs, np.linalg.matrix_rank(obs)          # Return the observability-matrix and it's rank

cont_matrix = cont(A, B, n)     # Calculate the controllability-matrix
obs_matrix = obs(A, C, n)       # Calculate the observability-matrix

print(f'Controllabillity matrix:\n {cont_matrix[0]} \n Rank of controllability matrix: {cont_matrix[1]}\n')
print(f'Observability matrix:\n {obs_matrix[0]} \n Rank of observability matrix: {obs_matrix[1]}\n')
