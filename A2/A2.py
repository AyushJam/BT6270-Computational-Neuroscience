"""
Assignment 2: BT6270 Conputational Neuroscience

@author: Ayush Jamdar EE20B018

Description:
    The goal of this assignment is to simulate the two-variable Fitzhugh-Nagumo model and analyse the simulation results.
    The FH model is a simplified neuron model derived from the more complex Hodgkin-Huxley model. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class FitzHugh_Nagumo_model:
    def __init__(self, a, b, r, Im):
        # model parameters
        self.a = a
        self.b = b
        self.r = r
        self.Im = Im
        self.vmax = 1.5
        self.vmin = -0.5
        self.wmax = Im + 0.4
        self.wmin = Im - 0.4

    def solve(self, v_init, w_init, t):
        """
        solves the FH differential equation using a simple Forward Euler

        input:
            v and w are initial values and t is the time array.
        output:
            no return
            data is stored in attributes
        """

        v = np.zeros(len(t))
        w = np.zeros(len(t))

        v[0] = v_init
        w[0] = w_init
        dt = t[1] - t[0]

        for i in range(1, len(t) - 1):
            dv = (
                v[i - 1] * (self.a - v[i - 1]) * (v[i - 1] - 1) - w[i - 1] + self.Im
            ) * dt
            v[i] = v[i - 1] + dv

            dw = (self.b * v[i - 1] - self.r * w[i - 1]) * dt
            w[i] = w[i - 1] + dw

        self.v_t = v
        self.w_t = w
        self.t_t = t

    def phase_plane_analysis(self):
        """
        Here, we solve for the attributes of phase plane
        This includes the nullclines and the vector field
        """

        num_pts = 25

        # v and w ranges
        v_grid = np.linspace(self.vmin, self.vmax, num_pts)
        w_grid = np.linspace(self.wmin, self.wmax, num_pts)

        V_grid, W_grid = np.meshgrid(v_grid, w_grid)

        V_dot = V_grid * (self.a - V_grid) * (V_grid - 1) - W_grid + self.Im
        W_dot = self.b * V_grid - self.r * W_grid

        self.V_grid = V_grid
        self.W_grid = W_grid
        self.V_dot = V_dot
        self.W_dot = W_dot

        # nullclines in meshgrid
        # vdot = 0 and wdot = 0
        v_nullcline = v_grid * (self.a - v_grid) * (v_grid - 1) + self.Im
        w_nullcline = (self.b * v_grid) / self.r
        # be careful while plotting (see phase_plot())

        self.v_nullcline = v_nullcline
        self.w_nullcline = w_nullcline

    def phase_plot(self, title="phase-plot.png"):
        """
        plot v-w phase from the meshgrid
        add nullclines to it on the same plot
        save the plot in directory
        """

        W_normalized = (self.W_grid - self.W_grid.min()) / (
            self.W_grid.max() - self.W_grid.min()
        )
        # Create a colormap
        cmap = plt.cm.viridis
        fig, ax = plt.subplots()
        ax.quiver(
            self.V_grid,
            self.W_grid,
            self.V_dot,
            self.W_dot,
            color=cmap(W_normalized.flatten()),
            scale=10,
        )
        # set limits on y axis
        ax.plot(self.V_grid[0], self.v_nullcline, label="v nullcline")
        ax.plot(self.V_grid[0], self.w_nullcline, label="w nullcline")
        ax.set_ylim([self.wmin, self.wmax])
        ax.set_xlabel("v")
        ax.set_ylabel("w")
        ax.set_title("Phase Plane Analysis")
        ax.legend()
        fig.savefig(title)

        self.phase_plane = ax
        return

    def time_plot(self, title="time-plot.png"):
        # plot v and w vs t
        # save the plot in directory
        plt.figure()
        plt.plot(self.t_t, self.v_t, label="v")
        plt.plot(self.t_t, self.w_t, label="w")
        plt.xlabel("t")
        plt.ylabel("v,w")
        plt.grid()
        plt.legend()
        plt.title("Time Plot")
        plt.savefig(title)
        return

    def phase_plane_trajectory(self, label="tj", title="phase-plane-trajectory.png"):
        # using the phase plane attribute of the object, plot the trajectory
        # note that the v and w ranges of the trajectory are different from that of the phase plot
        # save the plot in directory
        phase_plot = self.phase_plane
        phase_plot.plot(self.v_t[:-1], self.w_t[:-1], label=label)
        phase_plot.legend()
        phase_plot.set_title("Phase Plane Trajectory")
        phase_plot.figure.savefig(title)

        return


def find_oscillation_limits(a, b, r, time):
    """
    Find the limits on Im for which the system oscillates
    I1 < Im < I2
    Oscillations are measured in v(t)
    """

    # iterate over a range of Im values to get I1
    Imax = 1  # maximum value of current, to be used for array
    Imin = 0

    I_range = np.arange(Imin, Imax, 0.01)

    I1 = -1
    I2 = -1  # if -1, then no oscillations

    for i in I_range:
        model = FitzHugh_Nagumo_model(a, b, r, i)
        model.solve(0.4, 0, time)  # v_init = 0.4, w_init = 0
        v_t = model.v_t
        peaks, _ = find_peaks(v_t, height=0.45)
        if len(peaks) >= 3:
            model.time_plot("test_osc_1.png")
            I1 = i
            break

    for i in np.arange(I1, Imax, 0.01):
        model = FitzHugh_Nagumo_model(a, b, r, i)
        model.solve(0.4, 0, time)  # v_init = 0.4, w_init = 0
        v_t = model.v_t
        peaks, _ = find_peaks(v_t, prominence=0.4)
        if len(peaks) <= 2:
            model.time_plot("test_osc_2.png")
            I2 = i
            break

    return I1, I2


def case_1(a, b, r, time):
    # Case I: Im = 0
    fh_model_1 = FitzHugh_Nagumo_model(a, b, r, 0)

    # 1 (a) draw the phase plot
    fh_model_1.phase_plane_analysis()
    fh_model_1.phase_plot("phase-plot-1a.png")

    # 1 (b) solve the differential equation
    # (i) v(0) = a/2, w(0) = 0
    fh_model_1.solve(a / 2, 0, time)
    fh_model_1.time_plot("time-plot-1bi.png")
    fh_model_1.phase_plane_trajectory(label="trajectory-i", title="trajectory-1bi.png")

    # # (ii) v(0) = 2a, w(0) = 0
    fh_model_1.solve(2 * a, 0, time)
    fh_model_1.time_plot("time-plot-1bii.png")
    fh_model_1.phase_plane_trajectory(
        label="trajectory-ii", title="trajectory-1bii.png"
    )


def case_2(a, b, r, time, I1, I2):
    # Case II: I1 < Im < I2

    # 2 (a) draw the phase plot for Iext = mean(I1, I2)
    fh_model_2 = FitzHugh_Nagumo_model(a, b, r, (I1 + I2) / 2)
    fh_model_2.phase_plane_analysis()
    fh_model_2.phase_plot("phase-plot-2a.png")

    # 2 (c) solve the differential equation
    # v(0) = 0.4, w(0) = 0
    fh_model_2.solve(0.4, 0, time)
    fh_model_2.time_plot("time-plot-2c.png")
    fh_model_2.phase_plane_trajectory(label="trajectory", title="trajectory-2c.png")


def case_3(a, b, r, time, I1, I2):
    # Case III: Im > I2
    # 3 (a) draw the phase plot for Iext = I2 + 0.1
    Im = I2 + 0.1
    fh_model_3 = FitzHugh_Nagumo_model(a, b, r, Im)
    fh_model_3.phase_plane_analysis()
    fh_model_3.phase_plot("phase-plot-3a.png")

    # 3 (b) solve the differential equation
    # v(0) = 0.4, w(0) = 0
    fh_model_3.solve(0.4, 0, time)
    fh_model_3.time_plot("time-plot-3b.png")
    fh_model_3.phase_plane_trajectory(label="trajectory", title="trajectory-3b.png")


def case_4(a, time):
    # Find suitable values of Iext and b/r such that three points of intersection are found
    # Bistability
    # slope is b/r
    Im = 0.02
    b = 0.02
    r = 0.5
    fh_model_4 = FitzHugh_Nagumo_model(a, b, r, Im)
    fh_model_4.phase_plane_analysis()
    fh_model_4.phase_plot("phase-plot-4a.png")

    # 4 (b) solve the differential equation
    # v(0) = 0.4, w(0) = 0
    fh_model_4.solve(0.4, 0.05, time)
    fh_model_4.time_plot("time-plot-4b-ton-down.png")
    fh_model_4.phase_plane_trajectory(
        label="tj-tonically-down", title="trajectory-4b-ton-down.png"
    )

    # 4 (c) solve the differential equation
    # v(0) = 0.8, w(0) = 0
    fh_model_4.solve(0.8, -0.05, time)
    fh_model_4.time_plot("time-plot-4c-ton-up.png")
    fh_model_4.phase_plane_trajectory(
        label="tj-tonically-up", title="trajectory-4c-ton-up.png"
    )


def Assignment():
    # given model parameters
    a = 0.5
    b = 0.1
    r = 0.1

    time = np.linspace(0, 100, 1000)

    # Finding the values of I1 and I2
    I1, I2 = find_oscillation_limits(a, b, r, time)
    # print("I1 = {}, I2 = {}".format(I1, I2))

    case_1(a, b, r, time)
    case_2(a, b, r, time, I1, I2)
    case_3(a, b, r, time, I1, I2)
    case_4(a, time)

    return


# execute the assignment!
Assignment()
