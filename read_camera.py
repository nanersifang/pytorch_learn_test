import cv2
import skimage
from skimage.util.dtype import convert



import skimage
from skimage.segmentation import slic, mark_boundaries
from skimage import io,data
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import cv2

import torch as t
from densenet.dense_net import densenet, data_tf

from torchvision import transforms
from IP102.dataset_ip102 import Dataset_IP102

from Super_pixel import Super_pixel


device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

class_names = [x for x in range(102)] #这个顺序很重要，要和训练时候的类名顺序一致

#test_dataset = Dataset_IP102('f:/5.datasets/ip102_20201116/ip102_v1.1',train=False,transforms=data_tf)
test_dataset = Super_pixel('./data/camera',train=False,transforms=data_tf)
test_dataloader = t.utils.data.DataLoader(test_dataset,
                                     batch_size=10,#14
                                     shuffle=True,
                                     drop_last=True)



def save_pic(segments, img, path):

    lst = []

    maxn = max(segments.reshape(int(segments.shape[0] * segments.shape[1]), ))
    for i in range(1, maxn + 1):
        a = np.array(segments == i)
        #b = img * a
        #print(a)
        #b = img[:,:,:] * np.array([a,a,a]).reshape((480,640,3))
        b = img[:, :, 1] * a
        w, h = [], []
        for x in range(b.shape[0]):
            for y in range(b.shape[1]):
                if b[x][y] != 0:
                    w.append(x)
                    h.append(y)

        c = b[min(w):max(w), min(h):max(h)]
        c = c * 255
        d = c.reshape(c.shape[0], c.shape[1], 1)
        e = np.concatenate((d, d), axis=2)
        e = np.concatenate((e, d), axis=2)
        img2 = Image.fromarray(np.uint8(e))
        img2.save(path + '\\' + str(i) + '.png')
        lst.append(img2)
        print('已保存第' + str(i) + '张图片')

    return lst


vc = cv2.VideoCapture(0)
# 加载摄像头，进行读取，此API也可以加载本地的视频
if vc.isOpened():
    # 来保证读取成功
    oepn, frame = vc.read()
else:
    open = False

while open:

    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        #   将读取的照片信息进行颜色的转化 0为灰色，1为RGB
        # 其实为了在图像上做处理而预备的
        gray = cv2.cvtColor(frame, 1)

        # noise_img = skimage.util.random_noise(gray, mode="salt")
        #gray = skimage.util.random_noise(gray, mode='gaussian')
        # gray = cv2.GaussianBlur(gray, ksize=3)
        #cv2.imshow('result', gray)

        segments = slic(gray, n_segments=50, compactness=0.2, start_label=1)  # 进行SLIC分割
        out = mark_boundaries(gray, segments,color=(1,1,0))
        #out = out * 255  # io的灰度读取是归一化值，若读取彩色图片去掉该行
        img3 = Image.fromarray(np.uint8(out))
        #img3.show()
        cv2.imshow('result', out)
        lst = save_pic(segments, out, './data/camera')

        net = densenet(3, 102)
        net.load_state_dict(t.load('./data/densenet_2021-09-28.pkl'))

        '''
        img_path = './data/camera1/0.jpg'

        # （1）此处为使用PIL进行测试的代码
        transform_valid = transforms.Compose([
            transforms.Resize(((96, 96),2)),
            transforms.ToTensor()

        ]
        )
        img = Image.open(img_path)
        #img_ = transform_valid(img).unsqueeze(0)  # 拓展维度
        img_ = data_tf(img)

        ##（2）此处为使用opencv读取图像的测试代码，若使用opencv进行读取，将上面（1）注释掉即可。
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (56, 56))
        # img_ = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)/255

        #img_ = img_.to(device)
        '''
        dataiter = iter(test_dataloader)
        images, labels = dataiter.next()  # 一个batch返回4张图片
        outputs = net(images)
        print(outputs.shape)


        # 输出概率最大的类别
        for i in range(10):
            _, indices = t.max(outputs.data, 1)
            percentage = t.nn.functional.softmax(outputs.data, dim=1)[0] * 100
            perc = percentage[int(indices[i])].item()
            result = class_names[indices[i]]
            print('predicted:', result ,end=' ')


        # 直接1 ms进行更细图像  知道按esc

        if cv2.waitKey(1) & 0xFF == 27:
            break

vc.release()
cv2.destroyAllWindows()
cv2.imshow()


