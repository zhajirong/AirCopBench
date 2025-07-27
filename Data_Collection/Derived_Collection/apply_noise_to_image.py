import cv2
import numpy as np
import os

def add_gaussian_noise(image, mean=0, std=25):
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_and_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    num_pixels = image.shape[0] * image.shape[1]
    num_salt = int(num_pixels * amount / 2)
    num_pepper = int(num_pixels * amount / 2)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def apply_noise_mask(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    image = cv2.resize(image, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)
    noisy = add_gaussian_noise(image, mean=0, std=25)
    noisy = add_salt_and_pepper_noise(noisy, amount=0.05)
    cv2.imwrite(output_path, noisy)

if __name__ == "__main__":
    origin_dir = "/Users/starryyu/Documents/tinghuasummer/trian_s/ORigin"
    output_dir = "/Users/starryyu/Documents/tinghuasummer/trian_s/noise"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for fname in os.listdir(origin_dir):
        if fname.lower().endswith('.jpg'):
            input_path = os.path.join(origin_dir, fname)
            output_path = os.path.join(output_dir, fname[:-4] + '_noisy.jpg')
            try:
                apply_noise_mask(input_path, output_path)
                print(f"已处理: {fname} -> {output_path}")
            except Exception as e:
                print(f"处理 {fname} 时出错: {e}")
    print("全部处理完成！")
