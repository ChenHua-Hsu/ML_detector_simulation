import torch
import matplotlib.pyplot as plt

# Load the tensor
file_path = "/eos/user/c/chenhua/copy_tdsm_encoder_sweep16/sampling_result/sampling_20241210_1402_output/sample.pt"
data = torch.load(file_path)

data = data[0]
print(type(data))
print(len(data))

# Check the shape of the tensor
#print("Shape of the tensor:", len(data[0][1]))

# Convert the list to a PyTorch tensor if necessary
# if isinstance(data, list):
#     data = torch.tensor(data)

# Extract the desired slice [:, :, 1]
slice_data = data[:][:][0]
plot_data = slice_data.flatten()

# Check the shape of the slice
print("Shape of slice [:, :, 1]:", slice_data.shape)

# # 1. Plot a heatmap
# plt.figure(figsize=(10, 6))
# plt.imshow(slice_data.numpy(), aspect='auto', cmap='viridis')
# plt.colorbar(label='Value')
# plt.title("Heatmap of [:, :, 1]")
# plt.xlabel("Second Dimension")
# plt.ylabel("First Dimension")
# plt.show()

# # 2. Line plot for specific rows (optional)
# for i in range(0, slice_data.shape[0], 100):  # Plot every 100th row for clarity
#     plt.plot(slice_data[i].numpy(), label=f"Row {i}")
# plt.title("Line Plot of Rows from [:, :, 1]")
# plt.xlabel("Second Dimension Index")
# plt.ylabel("Value")
# plt.legend()
# plt.savefig("line_plot.png")

plt.hist(plot_data, bins=100)
plt.savefig("hist.png")