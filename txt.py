import os

img_path = r'C:\Users\Nicole\PycharmProjects\pythonProject1\ORACLE_MNIST_SRC\data\oracle-mnist-test-easy'


for i,img in enumerate(os.listdir(img_path)):

    src = os.path.join(r'C:\Users\Nicole\PycharmProjects\pythonProject1\ORACLE_MNIST_SRC\data\oracle-mnist-test-easy', img)  # 原先的图片名字
    label = os.path.basename(src).split('.')[0][-1]

    with open(r'C:\Users\Nicole\PycharmProjects\pythonProject1\ORACLE_MNIST_SRC\data\test.txt', 'a+') as f:
        f.write(src+' '+label+'\n')
    print("{}已写入，当前已经完成{}个路径".format(os.path.basename(src),i+1))