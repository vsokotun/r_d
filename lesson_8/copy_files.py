import os
import shutil

train_dir = "/Users/sokotun/.cache/kagglehub/datasets/jessicali9530/stanford-cars-dataset/versions/2/cars_train/cars_train"
test_dir = "/Users/sokotun/.cache/kagglehub/datasets/jessicali9530/stanford-cars-dataset/versions/2/cars_test/cars_test"
target_dir = "/Users/sokotun/training/r_d/lesson_9/cars"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

def copy_images_with_unique_names(source_dir, target_dir, prefix=""):
    for idx, filename in enumerate(os.listdir(source_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            source_path = os.path.join(source_dir, filename)
            new_filename = f"{prefix}_{idx}_{filename}"
            target_path = os.path.join(target_dir, new_filename)
            shutil.copy(source_path, target_path)
            print(f"Copied {filename} to {new_filename}")

copy_images_with_unique_names(train_dir, target_dir, prefix="train")

copy_images_with_unique_names(test_dir, target_dir, prefix="test")