import matplotlib.pyplot as plt
import numpy as np

# Read data from results.txt
with open("results.txt", "r") as f:
    data = f.readlines()

# Read data from results_without_Attention.txt
with open("results_without_Attention.txt", "r") as f:
    data_wo_attention = f.readlines()

# Initialize empty arrays to store data
loss = np.zeros((2, 21))
bright_mae = np.zeros((2, 21))
bright_mse = np.zeros((2, 21))
bright_psnr = np.zeros((2, 21))
bright_ssim = np.zeros((2, 21))
bright_ab = np.zeros((2, 21))

# Extract data from results.txt
for i in range(6):
    values = data[i].strip().split(",")
    for j in range(21):
        value = float(values[j])
        if i == 0:
            loss[0][j] = value
        elif i == 1:
            bright_mae[0][j] = value
        elif i == 2:
            bright_mse[0][j] = value
        elif i == 3:
            bright_psnr[0][j] = value
        elif i == 4:
            bright_ssim[0][j] = value
        elif i == 5:
            bright_ab[0][j] = value

# Extract data from results_without_Attention.txt
for i in range(6):
    values = data_wo_attention[i].strip().split(",")
    for j in range(21):
        value = float(values[j])
        if i == 0:
            loss[1][j] = value
        elif i == 1:
            bright_mae[1][j] = value
        elif i == 2:
            bright_mse[1][j] = value
        elif i == 3:
            bright_psnr[1][j] = value
        elif i == 4:
            bright_ssim[1][j] = value
        elif i == 5:
            bright_ab[1][j] = value

# Plot the graphs
x = np.arange(0, 21)

fig, ax1 = plt.subplots()
plt.figure(figsize=(8,6))
ax1.plot(x, loss[0], label="Network_with_Attention")
ax1.plot(x, loss[1], label="Network_without_Attention")
ax1.set_xticks(np.arange(min(x), max(x)+1, 1))
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
plt.show()

fig, ax2 = plt.subplots()
plt.figure(figsize=(8,6))
ax2.plot(x, bright_mae[0], label="Network_with_Attention")
ax2.plot(x, bright_mae[1], label="Network_without_Attention")
ax2.set_xticks(np.arange(min(x), max(x)+1, 1))
ax2.set_title("Bright MAE")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Bright MAE")
ax2.legend()
plt.show()

fig, ax3 = plt.subplots()
plt.figure(figsize=(8,6))
ax3.plot(x, bright_mse[0], label="Network_with_Attention")
ax3.plot(x, bright_mse[1], label="Network_without_Attention")
ax3.set_xticks(np.arange(min(x), max(x)+1, 1))
ax3.set_title("Bright MSE")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Bright MSE")
ax3.legend()
plt.show()

fig, ax4 = plt.subplots()
plt.figure(figsize=(8,6))
ax4.plot(x, bright_psnr[0], label="Network_with_Attention")
ax4.plot(x, bright_psnr[1], label="Network_without_Attention")
ax4.set_xticks(np.arange(min(x), max(x)+1, 1))
ax4.set_title("Bright PSNR")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Bright PSNR")
ax4.legend()
plt.show()

fig, ax5 = plt.subplots()
plt.figure(figsize=(8,6))
ax5.plot(x, bright_ssim[0], label="Network_with_Attention")
ax5.plot(x, bright_ssim[1], label="Network_without_Attention")
ax5.set_xticks(np.arange(min(x), max(x)+1, 1))
ax5.set_title("Bright SSIM")
ax5.set_xlabel("Epoch")
ax5.set_ylabel("Bright SSIM")
ax5.legend()
plt.show()

fig, ax6 = plt.subplots()
plt.figure(figsize=(8,6))
ax6.plot(x, bright_ab[0], label="Network_with_Attention")
ax6.plot(x, bright_ab[1], label="Network_without_Attention")
ax6.set_xticks(np.arange(min(x), max(x)+1, 1))
ax6.set_title("Bright AB")
ax6.set_xlabel("Epoch")
ax6.set_ylabel("Bright AB")
ax6.legend()
plt.show()
