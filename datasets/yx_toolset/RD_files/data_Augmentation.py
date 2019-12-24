#随机crop
#就是把图像加减某个值
# 2. 随机变换亮度 (概率：0.5)
    def random_bright(self, im, delta=32):
        if random.random() < 0.5:
            delta = random.uniform(-delta, delta)
            im += delta
            im = im.clip(min=0, max=255)
        return im
'''
其中：
* random.uniform(-a, a) 随机在-a到a之间生成数字
* np.clip(min，max) 和tensor.clamp效果一样
'''
# 3随机变换通道
def random_swap(self, im):
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    if random.random() < 0.5:
        swap = perms[random.randrange(0, len(perms))]
        im = im[:, :, swap]
    return im
    
#图片像素点随机乘以某个值，（0.5,1.5）
# 4. 随机变换对比度
def random_contrast(self, im, lower=0.5, upper=1.5):
    if random.random() < 0.5:
        alpha = random.uniform(lower, upper)
        im *= alpha
        im = im.clip(min=0, max=255)
    return im
#就是中间的色彩通道乘以某个值
# 5. 随机变换饱和度
    def random_saturation(self, im, lower=0.5, upper=1.5):
        if random.random() < 0.5:
            im[:, :, 1] *= random.uniform(lower, upper)
        return im
        
#6随机变换色度(HSV空间下(-180, 180))
   def random_hue(self, im, delta=18.0):
       if random.random() < 0.5:
           im[:, :, 0] += random.uniform(-delta, delta)
           im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
           im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
       return im
#随机饱和度和色度变换

#变换HSV空间，然后变回到OpenCV的BGR空间
#1数据增强过程，取出每个500不同的图片做数据增强2-6
#2做完数据增强，把图片crop成模式输入大学- 800 - 1280
#3训练设置参数调节 anchor batch——size lr等
#4 优化器的选择最近用Adam多一些，loss的重定义采用Focal loss mxnet框架应该有封装好的api需要查阅
#5最后进阶把faster cnn 和FPN+DCN融合利用rpn技术提升精度

'''
    如果有时间可以做这些策略下面简要讲解一目标增强的策略：
    子策略1 沿X轴方向的平移，发生概率0.6，程度为-30像素（文中将平移的程度范围[-150 pixels, 150 pixels]对应到[0, 10]，对应地，4代表-30像素，依此类推）；均衡化，发生概率为0.8（由于直方图均衡化并没有程度，所以只有发生概率）
    子策略2 检测框内像素沿Y轴方向的平移，发生概率为0.2，程度为-90像素；cutout，随机从图像中减去一个正方形区域（置灰），发生概率为0.8，程度为48像素
    子策略3 shear，让所有点x轴不动，y轴坐标按比例平移，即x = x0，y = y0 + k*x0，发生概率为1.0，程度为-0.18，即k = -0.18；检测框内像素沿Y轴方向的平移，发生概率为0.6，程度为30像素
    子策略4 旋转，发生概率为0.6，程度为30度；color，调整色彩平衡，发生概率为1.0，程度为1.18
    子策略5 空操作；空操作
    对于上述5个子策略，对于mini-batch里的每个图像都会随机从5个子策略中选择一个，
'''
