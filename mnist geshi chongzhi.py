import os
import cv2
import numpy as np
#原有的mnist格式很奇怪，这个文件可以转换为png格式

def save_mnist_to_jpg(dataset_path, output_path):  # parameters：（数据集所在路径，最后输出存储路径）
    files = os.listdir(dataset_path)
    # 生成image_file路径
    mnist_image_file = os.path.join(dataset_path, [f for f in files if "image" in f][0])
    # 生成label_file路径
    mnist_label_file = os.path.join(dataset_path, [f for f in files if "label" in f][0])
    save_dir = output_path
    num_file = 27222
    height, width = 28, 28
    size = height * width
    prefix = 'test'
    # 二进制形式读取文件
    with open(mnist_image_file, 'rb') as f1:
        image_file = f1.read()
    with open(mnist_label_file, 'rb') as f2:
        label_file = f2.read()
    image_file = image_file[16:]  # 将所读数据进行切片，去掉开头16字节
    label_file = label_file[8:]
    for i in range(num_file):
        label = label_file[i]
        image_list = [item for item in image_file[i * size:i * size + size]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(height, width)
        save_name = os.path.join(save_dir, '{}_{}_{}.jpg'.format(prefix, i, label))
        cv2.imwrite(save_name, image_np)
    print("=" * 20, "preprocess data finished", "=" * 20)


if __name__ == '__main__':
    save_mnist_to_jpg(r'/oracle-mnist-train', r'/ORACLE_MNIST_SRC/data/oracle-mnist-train-easy')