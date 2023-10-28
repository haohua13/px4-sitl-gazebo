import numpy as np
import matplotlib.pyplot as plt
# Load the reshaped arrays
W_save_reshaped = np.load('W_save_reshaped.npy')
q_save_reshaped = np.load('q_save_reshaped.npy')
# Plot the data
plt.figure(figsize=(10, 5))
# Plot W_save
plt.subplot(1, 2, 1)
plt.plot(W_save_reshaped)
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Translational Optical Flow (W_save)')
plt.legend(['X', 'Y', 'Z'])


# Plot q_save
plt.subplot(1, 2, 2)
plt.plot(q_save_reshaped)
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Position Measurements (q_save)')
plt.legend(['X', 'Y', 'Z'])

plt.tight_layout()
plt.show()

from scipy.spatial.transform import Rotation as R

R = R.from_euler('XYZ', [0, 0, 0], degrees = True)
rotation_matrix = np.array(R.as_matrix())
print(rotation_matrix)

import scipy

scipy.io.savemat('q.mat', mdict={'q': q_save_reshaped})
scipy.io.savemat('W.mat', mdict={'W': W_save_reshaped})

