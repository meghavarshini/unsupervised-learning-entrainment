import h5py


# Open the HDF5 file in read mode
file_path = "./multicat_h5_output/test_ASIST.h5"  # Replace with the actual file path
with h5py.File(file_path, "r") as h5f:
    # def print_structure(name, obj):
    #     print(name)  # Print the name of the group/dataset
    # h5f.visititems(print_structure)

    print("Keys in the file:", list(h5f.keys()))  # Lists the top-level groups
    for key in h5f.keys():
        print(f"Dataset {key}: shape {h5f[key].shape}, dtype {h5f[key].dtype}")

    # data = h5f[dataset_name][()]
    # print("Data:", data)  # Print data (be careful with large datasets)
