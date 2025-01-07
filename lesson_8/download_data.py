import kagglehub

# Download latest version
path = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")

print("Path to dataset files:", path)


# Download latest version
compcars_path = kagglehub.dataset_download("renancostaalencar/compcars")

print("Path to dataset files:", compcars_path)