import os
import torch
from torchvision import transforms as T
from PIL import Image

data_dir = "cars"
compcars_dir = "/Users/sokotun/.cache/kagglehub/datasets/renancostaalencar/compcars/versions/1"
save_path = "processed_data.pt"
img_h = 48
img_w = 64

transform = T.Compose([
    T.Resize((img_h, img_w)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_all_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths

def prepare_dataset():
    image_paths = get_all_image_paths(data_dir)
    
    compcars_images_dir = os.path.join(compcars_dir, "image")
    compcars_paths = get_all_image_paths(compcars_images_dir)
    
    print(f"Files in {data_dir}: {len(image_paths)}")
    print(f"Files in {compcars_dir}: {len(compcars_paths)}")
    print(f"Total files: {len(image_paths) + len(compcars_paths)}")
    
    images = []
    
    print("Загрузка и преобразование изображений...")
    for img_path in image_paths + compcars_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)
    
    images = torch.stack(images)
    
    print("Сохранение обработанных данных...")
    torch.save(images, save_path)
    print(f"Данные сохранены в {save_path}")

if __name__ == "__main__":
    prepare_dataset() 