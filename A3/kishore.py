# Importing required Libraries
import numpy as np
import matplotlib.pyplot as plt

# Part 1 : Complex Coupling

# Defining the time array
tmax = 20
dt = 0.01
t = np.arange(0, tmax, dt)

# Defining the parameters
w = 5   # w1 = w2 = w
phi = [-47*(np.pi/180), 98*(np.pi/180)]  # Converting to radians
r1 = np.zeros_like(t)
theta1 = np.zeros_like(t)
r2 = np.zeros_like(t)
theta2 = np.zeros_like(t)
# Intital values
r1[0] = 1
r2[0] = 0.5

def cc_dr1dt(A, phi, r1, theta1, r2, theta2):
    mu = 1
    dr1dt = (mu-r1**2)*r1 + A*r2*np.cos(theta2 - theta1 + phi)
    return dr1dt

def cc_dtheta1dt(A, w1, phi, r1, theta1, r2, theta2):
    dtheta1dt = w1 + A*r2*np.sin(theta2 - theta1 + phi)/r1
    return dtheta1dt

def cc_dr2dt(A, phi, r1, theta1, r2, theta2):
    mu = 1
    dr2dt = (mu-r2**2)*r2 + A*r1*np.cos(theta1 - theta2 - phi)
    return dr2dt

def cc_dtheta2dt(A, w2, phi, r1, theta1, r2, theta2):
    dtheta2dt = w2 + A*r1*np.sin(theta1 - theta2 - phi)/r2
    return dtheta2dt

def complex_coupling(r1, theta1, r2, theta2, w, phi):
    # Updating the variables
    for time in range(len(t)-1):
        r1[time+1] = r1[time] + (cc_dr1dt(0.2, phi, r1[time], theta1[time], r2[time], theta2[time]))*dt
        theta1[time+1] = theta1[time] + (cc_dtheta1dt(0.2, w, phi, r1[time], theta1[time], r2[time], theta2[time]))*dt
        r2[time+1] = r2[time] + (cc_dr2dt(0.2, phi, r1[time], theta1[time], r2[time], theta2[time]))*dt
        theta2[time+1] = theta2[time] + (cc_dtheta2dt(0.2, w, phi, r1[time], theta1[time], r2[time], theta2[time]))*dt
    # Plotting the real parts
    plt.plot(t, r1*np.cos(theta1), label = 'Osc1')
    plt.plot(t, r2*np.cos(theta2), label = 'Osc2')
    plt.title('Real parts')
    plt.legend()
    plt.show()
    # Plotting the phase
    plt.plot(t, (theta1-theta2)*(180/np.pi))
    plt.title('Phase difference (in deg) between the oscillators vs time')
    plt.show()


# Calling the defined functions
# complex_coupling(r1, theta1, r2, theta2, w, phi[0])
# complex_coupling(r1, theta1, r2, theta2, w, phi[1])

# Part 2 : Power Coupling

# Defining the time array
tmax = 100
dt = 0.01
t = np.arange(0, tmax, dt)

# Defining the parameters
w1 = 5
w2 = 15
phi = [-47*(np.pi/180), 98*(np.pi/180)] # Converting to radians
r1 = np.zeros_like(t)
theta1 = np.zeros_like(t)
r2 = np.zeros_like(t)
theta2 = np.zeros_like(t)
# Storing derivatives also, Required for plotting
r1dot = np.zeros_like(t)
theta1dot = np.zeros_like(t)
r2dot = np.zeros_like(t)
theta2dot = np.zeros_like(t)
# Intital values
r1[0] = 1
r2[0] = 0.5
theta1[0] = np.pi
theta2[0] = 0
r1dot[0] = 0
r2dot[0] = 0
theta1dot[0] = 0
theta2dot[0] = 0

def pc_dr1dt(A, w1, w2, phi, r1, theta1, r2, theta2):
    mu = 1
    dr1dt = (mu-r1**2)*r1 + A*(r2**(w1/w2))*np.cos(w1*((theta2/w2) - (theta1/w1) + (phi/w1*w2)))
    return dr1dt

def pc_dtheta1dt(A, w1, w2, phi, r1, theta1, r2, theta2):
    dtheta1dt = w1 + A*(r2**(w1/w2))*np.sin(w1*((theta2/w2) - (theta1/w1) + (phi/w1*w2)))/r1
    return dtheta1dt

def pc_dr2dt(A, w1, w2, phi, r1, theta1, r2, theta2):
    mu = 1
    dr2dt = (mu-r2**2)*r2 + A*(r1**(w2/w1))*np.cos(w2*((theta1/w1) - (theta2/w2) - (phi/w1*w2)))
    return dr2dt

def pc_dtheta2dt(A, w1, w2, phi, r1, theta1, r2, theta2):
    dtheta2dt = w2 + A*(r1**(w2/w1))*np.sin(w2*((theta1/w1) - (theta2/w2) - (phi/w1*w2)))/r2
    return dtheta2dt

def power_coupling(r1, theta1, r2, theta2, w1, w2, phi):
    # Updating the variables
    for time in range(len(t)-1):
        A = 0.2
        r1dot[time] = pc_dr1dt(A, w1, w2, phi, r1[time], theta1[time], r2[time], theta2[time])
        theta1dot[time] = pc_dtheta1dt(A, w1, w2, phi, r1[time], theta1[time], r2[time], theta2[time])
        r2dot[time] = pc_dr2dt(A, w1, w2, phi, r1[time], theta1[time], r2[time], theta2[time])
        theta2dot[time] = pc_dtheta2dt(A, w1, w2, phi, r1[time], theta1[time], r2[time], theta2[time])
        r1[time+1] = r1[time] + r1dot[time]*dt
        theta1[time+1] = theta1[time] + theta1dot[time]*dt
        r2[time+1] = r2[time] + r2dot[time]*dt
        theta2[time+1] = theta2[time] + theta2dot[time]*dt

    # Plotting the real parts
    # plt.plot(t, r1*np.cos(theta1), label = 'Osc1')
    # plt.plot(t, r2*np.cos(theta2), label = 'Osc2')
    # plt.title('Real parts')
    # plt.legend()
    # plt.xlim(0,20)
    # plt.show()

    # Plotting the normalized phase difference [should go to zero]
    plt.plot(t, ((theta1/w1)- (theta2/w2) - phi))
    plt.title('Normalized Phase difference (in deg) between the oscillators vs time')
    plt.show()

    # Plotting the normalized phase and it's time derivative
    plt.plot(t, ((theta1/w1)-(theta2/w2)), label = 'psi')
    plt.plot(t, ((theta1dot/w1)-(theta2dot/w2)), label = 'psidot')
    plt.title('$\psi$ and d$\psi$/dt vs time')
    plt.legend()
    plt.show()

# Calling the defined functions
power_coupling(r1, theta1, r2, theta2, w1, w2, phi[0])
# power_coupling(r1, theta1, r2, theta2, w1, w2, phi[1])