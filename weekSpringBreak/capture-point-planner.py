import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
from matplotlib.widgets import Slider


def generate_footsteps(num_steps: int, step_length: float, step_width:float) -> np.ndarray:
    # Stored such that each row is a footstep x and y, col 1 = x, col 2 = y.
    footstep_locations = np.zeros((num_steps, 2))
    # In the x (side-to-side) direction, we want steps to oscillate back and forth with step_width.
    footstep_locations[:, 0] = step_width
    footstep_locations[1::2, 0] = -step_width
    # In the y (forward) direction, we want steps to increase with step_length.
    for i in range(1, num_steps):
        footstep_locations[i, 1] = step_length * i
    return footstep_locations


def generate_dcm_reference(footstep_locations: np.ndarray, omega: float, step_time: float) -> np.ndarray:
    num_steps = footstep_locations.shape[0]
    dcm_keypoints = np.zeros((num_steps, 2))
    # Because the dcm_keypoints are found by working from the end of the footstep sequence backwards, we'll flip
    # the footstep_locations back-to-front so the for loop is nicer.
    footstep_locations = np.flip(footstep_locations, 0)
    dcm_keypoints[0, :] = footstep_locations[0, :]
    for i in range(1, num_steps):
        dcm_keypoints[i, :] = footstep_locations[i, :] + np.exp(-omega * step_time) * (dcm_keypoints[i-1, :] - footstep_locations[i, :])
    # We've then got to flip our dcm_keypoints to get them in the correct order.
    dcm_keypoints = np.flip(dcm_keypoints, 0)
    return dcm_keypoints


def interleave_rows(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    assert array1.shape == array2.shape
    shape = array1.shape
    result = np.zeros((2 * shape[0], shape[1]))
    for i in range(0, shape[0]):
        result[2 * i, :] = array1[i, :]
        result[2 * i + 1, :] = array2[i, :]
    return result


def explicit_euler(A: np.ndarray, B: np.ndarray, delta_t: float):
    A_d = A * delta_t + np.eye(A.shape[0])
    B_d = B * delta_t
    return A_d, B_d

def generate_system_matrices(omega, delta_t):
    # stacked x-direction and y-direction systems, e.g. state = [x, y, eta_x, eta_y]
    A = np.array([[-omega, 0.0, omega, 0.0],
                  [0.0, -omega, 0.0, omega],
                  [0.0, 0.0, omega, 0.0],
                  [0.0, 0.0, 0.0, omega]])
    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [-omega, 0.0],
                  [0.0, -omega]])
    A_d, B_d = explicit_euler(A, B, delta_t)
    return A_d, B_d


def generate_footstep_box_collection(footstep_locations):
    footstep_boxes = []
    footstep_box_width = 0.1
    footstep_box_height = 0.3
    for i in range(num_steps):
        footstep_box = mpatches.Rectangle(
            (footstep_locations[i, 0] - (footstep_box_width/2), footstep_locations[i, 1] - (footstep_box_height/2)),
            footstep_box_width, footstep_box_height)
        footstep_boxes.append(footstep_box)
    footstep_box_collection = PatchCollection(footstep_boxes, color='b', alpha=0.3)
    return footstep_box_collection


# Constants
g = 9.81  # gravity
zc = 0.8  # height of pelvis
omega = np.sqrt(g / zc)

# Modifiable
step_length = 0.6
step_width = 0.2
num_steps = 6
step_time = 0.5  # step time in seconds

def main(step_length, step_width, num_steps, step_time):
    footsteps = generate_footsteps(num_steps, step_length, step_width)
    dcm_keypoints = generate_dcm_reference(footsteps, omega, step_time)

    delta_t = 0.01
    t0 = 0.0
    tf = 100.0
    A_d, B_d = generate_system_matrices(omega, delta_t)

    x0 = np.array([0.0, 0.0])
    eta0 = dcm_keypoints[0]
    state0 = np.hstack((x0, eta0))

    # main loop
    t = t0
    state = state0
    state_tape = state0
    eta_d_tape = state0[2:]
    control_tape = footsteps[0]
    step_number = 0
    inter_step_time = 0.0
    while t < tf:
        if t > (step_number + 1) * step_time:
            step_number += 1
            inter_step_time = 0.0
        if step_number == num_steps-1:
            break  # Exit condition
        eta_d = footsteps[step_number] + np.exp(omega * (inter_step_time - step_time)) * (dcm_keypoints[step_number+1] - footsteps[step_number])
        eta_d_tape = np.vstack((eta_d_tape, eta_d))
        k = 1.0
        eta = state[2:]
        control = footsteps[step_number] + (1 + k/omega) * (eta - eta_d)
        control_tape = np.vstack((control_tape, control))
        state = A_d @ state + B_d @ control
        state_tape = np.vstack((state_tape, state))
        inter_step_time += delta_t
        t += delta_t

    return footsteps, dcm_keypoints, state_tape


footsteps, dcm_keypoints, state_tape = main(step_length, step_width, num_steps, step_time)

# Plotting
fig, ax = plt.subplots(1)
ax.grid()
ax.plot(footsteps[:, 0], footsteps[:, 1], 'ob')  # Footstep Locations
ax.add_collection(generate_footstep_box_collection(footsteps))  # Footstep prints
ax.plot(dcm_keypoints[:, 0], dcm_keypoints[:, 1], 'xr--')  # DCM keypoints
# Lines from DCM keypoints to footsteps
dcm_to_footstep_lines = interleave_rows(footsteps, dcm_keypoints)
for i in range(0, (2*num_steps-1), 2):
    ax.plot(dcm_to_footstep_lines[i:i+2, 0], dcm_to_footstep_lines[i:i+2, 1], ',k:')
ax.plot(state_tape[0, 1], state_tape[0, 1], 'xm')  # Initial CoM
ax.plot(state_tape[0, 2], state_tape[0, 3], '*g')  # Initial DCM
ax.plot(state_tape[:, 0], state_tape[:, 1], 'm')  # COM evolution
ax.plot(state_tape[:, 2], state_tape[:, 3], 'g')  # DCM evolution
# ax.plot(eta_d_tape[:, 0], eta_d_tape[:, 1], 'y')  # Desired DCM evolution
ax.set_xlim([-1, 1])
plt.show()

# State
# plt.plot(state_tape[:, 0])
# plt.plot(state_tape[:, 1])
# plt.plot(state_tape[:, 2])
# plt.plot(state_tape[:, 3])
# plt.grid()
# plt.show()

# Control
# plt.plot(control_tape[:, 0])
# plt.plot(control_tape[:, 1])
# plt.grid()
# plt.show()


