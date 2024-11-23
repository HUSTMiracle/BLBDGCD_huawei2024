from PIL import Image, ImageEnhance, ImageOps
import math
import os
import cv2
import numpy as np

class image:
    def __init__(self, img=None, x=0, y=0, x1=0, y1=0, error_list=None):
        """
        初始化对象的构造函数。

        此构造函数用于创建一个对象实例，并初始化其属性值。这些属性包括图像对象及其在图像坐标系中的起始和结束点，以及一个 error_list 属性用于存储额外的错误信息。

        参数:
        img - 图像对象，默认为None，表示没有指定图像。
        x - 图像的起始x坐标，默认为0。
        y - 图像的起始y坐标，默认为0。
        x1 - 图像的结束x坐标，默认为0。
        y1 - 图像的结束y坐标，默认为0。
        error_list - 储存错误信息的列表，默认为None，如果未提供，则初始化为空列表。

        返回值:
        无
        """
        # 初始化对象的属性
        self.img = img  # 图像对象
        self.x = x      # 图像起始点的x坐标
        self.y = y      # 图像起始点的y坐标
        self.x1 = x1    # 图像结束点的x坐标
        self.y1 = y1    # 图像结束点的y坐标
        if error_list is None:
            error_list = []  # 如果没有提供error_list，初始化一个新的空列表
        self.error_list = error_list  # 储存错误信息


def Img_Segmentation(img, n, overlap_rate):
    """
    对图像进行分割，返回分割后的图像列表。

    参数:
    - img: 需要分割的图像。
    - n: 图像分割的单元格数量，表示将图像划分为n*n个单元格。
    - overlap_rate: 分割后的图像重叠率。

    返回值:
    - img_List: 分割后的图像列表，每个元素为分割出的一块图像及其在原图中的位置信息。
    """

    def corner_case(xi, yi, wide, high, length, width, height, addlength, n):
        """
        计算并返回给定单元格坐标和尺寸参数对应的矩形的左下角和右上角坐标。

        参数:
        - xi: 单元格的x索引
        - yi: 单元格的y索引
        - wide: 单元格宽度
        - high: 单元格高度
        - length: 整个区域的长度（x方向）
        - width: 整个区域的宽度（y方向）
        - height: 整个区域的高度
        - addlength: 每个单元格额外增加的长度
        - n: 单元格总数（行或列）

        返回值:
        - x, y: 矩形的左下角坐标
        - x1, y1: 矩形的右上角坐标
        """
        # 计算单元格左下角坐标，考虑了额外增加的长度
        x = xi * wide - (addlength / 2 if xi != n - 1 else addlength)
        y = yi * high - (addlength / 2 if yi != n - 1 else addlength)
        # 确保x, y不为负数
        x = max(x, 0)
        y = max(y, 0)
        # 计算单元格右上角坐标，考虑了额外增加的长度
        x1 = (xi + 1) * wide + (addlength / 2 if xi != 0 else addlength)
        y1 = (yi + 1) * high + (addlength / 2 if yi != 0 else addlength)
        # 确保x1, y1不超过整个区域的尺寸
        x1 = min(x1, width)
        y1 = min(y1, height)
        return x, y, x1, y1

    img_List = []
    # 检查重叠率是否合法
    if overlap_rate > 1:
        print("Value Error")
        img_List.append(img)
        return img_List

    # 获取图像尺寸
    width, height = img.size[:2]
    # 计算分割后每张图像的基准尺寸
    height_base = height / n
    width_base = width / n

    # 计算由于重叠导致的额外长度
    add_length = (math.sqrt((height + width) ** 2 + 4 * height * width * overlap_rate) - (height + width)) / (2 * n)

    # 打印计算的基本参数，用于调试
    print(height_base, width_base, add_length)
    # 遍历所有分割块，进行图像分割
    for i in range(1, n * n + 1):
        xi, yi = (i - 1) % n, (i - 1) // n

        # 处理边界情况
        if xi == 0 or xi == n - 1 or yi == 0 or yi == n - 1:
            x, y, x1, y1 = corner_case(xi, yi, width_base, height_base, i, width, height, add_length, n)
        else:
            # 计算非边界分割块的坐标
            x = width_base * xi - add_length / 2
            y = height_base * yi - add_length / 2
            x1 = width_base * (xi + 1) + add_length / 2
            y1 = height_base * (yi + 1) + add_length / 2
        # 进行图像分割，并添加到结果列表中
        img_seg = img.crop((x, y, x1, y1))
        image_seg = image(img_seg, x, y, x1, y1)
        img_List.append(image_seg)

    return img_List


def square_img_segmentation(img, length, width, overlap_rate):
    """
    对图像进行正方形分割，返回分割后的图像列表。

    参数:
    - img: 需要分割的图像。
    - length: 图像大小。
    - width: 图像大小。
    - overlap_rate: 分割后的图像重叠率。

    返回值:
    - img_List: 分割后的图像列表，每个元素为分割出的一块图像及其在原图中的位置信息。
    """
    assert length == width, "The length and width of the image must be equal."
    assert overlap_rate <= 1, "The overlap rate must be less than or equal to 1."
    img_List = []
    l = img.size[0]
    w = img.size[1]
    current_l = 0
    current_w = 0
    flag = True
    while flag:
        x = current_l
        y = current_w
        x1 = current_l + length
        y1 = current_w + width
        if x1 > l and y1 > w:
            flag = False
        if x1 > l:
            x = l - length
            x1 = l
        if y1 > w:
            y = w - width
            y1 = w
        img_seg = img.crop((x, y, x1, y1))
        image_seg = image(img_seg, x, y, x1, y1)
        img_List.append(image_seg)
        if current_l + length >= l:
            current_l = 0
            if current_w + width >= w:
                flag = False
            current_w += width * (1 - overlap_rate)
        else:
            current_l += length * (1 - overlap_rate)

    return img_List


def update_label(img, segment_img, label_path):
    """
    根据提供的图像、分割图像和标签路径，更新标签的位置并返回更新后的错误列表。

    参数:
    img: 原始图像对象，具有width和height属性，用于更新标签位置时的尺寸参考。
    segment_img: 分割图像对象，具有x, y, x1, y1属性定义图像在原始图像中的位置，以及error_list属性用于存储更新后的标签位置信息。
    label_path: 字符串，表示标签文件的路径，文件中每一行包含标签的原始位置和尺寸信息。

    返回值:
    返回一个列表，包含所有更新后位于分割图像内的标签的新位置信息。
    """

    class LabelAdjuster:
        def __init__(self, label_path):
            """
            初始化标签调整器，读取并存储标签文件的内容。

            参数:
            label_path: 字符串，表示标签文件的路径。
            """
            self.labels_List = self.read_label(label_path)

        def read_label(self, label_path):
            """
            从给定的文件路径读取标签信息，并存储为列表。

            参数:
            label_path: 字符串，表示标签文件的路径。

            返回值:
            返回一个二维列表，每个子列表代表一个标签的位置和尺寸信息。
            """
            label_list = []  # 初始化存储标签的列表
            # 打开并读取标签文件内容
            with open(label_path, 'r') as file:
                for line in file:
                    # 将每行文本转换为整数列表
                    number_list = [float(number) for number in line.split()]
                    label_list.append(number_list)  # 将转换后的整数列表添加到标签列表中

            return label_list

        def update(self, segment_img, label_list):
            """
            根据原始图像尺寸更新标签位置，并将更新后的位于分割图像内的标签添加到分割图像的错误列表中。

            参数:
            segment_img: 分割图像对象，具有定义图像在原始图像中位置的属性和存储错误列表的属性。
            label_list: 二维列表，包含标签的位置和尺寸信息。

            返回值:
            返回分割图像对象的错误列表，包含更新后的标签位置信息。
            """
            # 遍历二维列表的每一行，并对元素进行处理
            length = segment_img.x1 - segment_img.x
            height = segment_img.y1 - segment_img.y
            for row_index, row in enumerate(label_list):
                # 更新奇数索引位置的元素值（假设为x坐标）基于原始图像的宽度
                for col_index in range(1, len(row), 2):
                    label_list[row_index][col_index] *= img.width
                # 更新偶数索引位置的元素值（假设为y坐标）基于原始图像的高度
                for col_index in range(2, len(row), 2):
                    label_list[row_index][col_index] *= img.height
            for row in label_list:
                # 判断标签是否位于当前分割图像内
                if row[1] > segment_img.x and row[1] < segment_img.x1 and row[2] > segment_img.y and row[2] < segment_img.y1:
                    # 计算标签在分割图像内的新位置，并更新到错误列表中
                    update_label_list = [int(row[0]), (row[1] - segment_img.x)/length, (row[2] - segment_img.y)/height,
                                         row[3]/length, row[4]/height]
                    segment_img.error_list.append(update_label_list)
            return segment_img.error_list
    adjuster = LabelAdjuster(label_path)
    return adjuster.update(segment_img, adjuster.labels_List)


def todarkblue(img):
    # 确保图片模式为RGB，如果是其他模式如RGBA，可以使用convert转换
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 获取图片的宽度和高度
    width, height = img.size

    # 遍历图片的每一个像素
    for x in range(width):
        for y in range(height):
            # 获取当前像素的RGB值
            r, g, b = img.getpixel((x, y))

            # 显示当前像素的RGB值
            # print(f"Pixel at ({x}, {y}) - RGB: ({r}, {g}, {b})")

            rgblist = [r, g, b]
            rgblist.sort()

            if r + g + b < 600:
                for i in range(3):
                    rgblist[i] = int((rgblist[i] * 0.8) ** 0.9)
                    if rgblist[i] > 255:
                        rgblist[i] = 255
            else:
                rgblist[2] -= 50

            # 修改像素的RGB值，这里简单地将每个通道值增加50作为示例
            new_r = rgblist[0]  # 确保修改后的值不超过255
            new_g = rgblist[1]
            new_b = rgblist[2]

            # 将新像素值设置回去
            img.putpixel((x, y), (new_r, new_g, new_b))

    return img


def tocolor(img, rr, gr, br):
    # 确保图片模式为RGB，如果是其他模式如RGBA，可以使用convert转换
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 获取图片的宽度和高度
    width, height = img.size

    # 遍历图片的每一个像素
    for x in range(width):
        for y in range(height):
            # 获取当前像素的RGB值
            r, g, b = img.getpixel((x, y))

            # 显示当前像素的RGB值
            # print(f"Pixel at ({x}, {y}) - RGB: ({r}, {g}, {b})")

            rgblist = [r, g, b]
            rgblist.sort()

            # 修改像素的RGB值，这里简单地将每个通道值增加50作为示例
            new_r = rgblist[rr]  # 确保修改后的值不超过255
            new_g = rgblist[gr]
            new_b = rgblist[br]

            # 将新像素值设置回去
            img.putpixel((x, y), (new_r, new_g, new_b))

    return img

def togreen(img):
    return tocolor(img, 0, 2, 2)

def toblue(img):
    return tocolor(img, 0, 1, 2)

def toorange(img):
    return tocolor(img, 2, 1, 0)

def tored(img):
    return tocolor(img, 2, 0, 0)

def topurple(img):
    return tocolor(img, 2, 0, 1)


def rotate_image(img_path, txt_path, angle=-1, save_img=None, save_txt=None):
    img = Image.open(img_path)
    with open(txt_path, 'r') as t:
        lines = t.readlines()
    if angle == -1:
        for __ in range(3):
            img = img.rotate(90)
            output_path = img_path.replace('.bmp', '') + f'_rotate{__}.jpg'
            img.save(output_path)
            for i in range(len(lines)):
                if lines[i][0] not in '0123456789':
                    continue
                line = lines[i].split(' ')
                temp = line[2]
                line[2] = "{:6f}".format(1 - float(line[1]))
                line[1] = temp
                temp = line[4][:-1]
                line[4] = line[3]
                line[3] = temp
                lines[i] = ' '.join(line) + '\n'

            with open(txt_path.replace('.txt', '') + f'_rotate{__}.txt', 'w') as t:
                t.writelines(lines)
    if save_img is not None and save_txt is not None and angle != -1:
        if angle == 0:
            img.save(save_img)
            with open(save_txt, 'w') as t:
                t.writelines(lines)
            return
        assert angle in [90, 180, 270]
        for __ in range(3):
            img = img.rotate(90)
            if __ * 90 == angle - 90:
                img.save(save_img)
            for i in range(len(lines)):
                # lines[i] = int(lines[i])
                # if lines[i][0] not in '0123456789':
                # continue
                lines[i].replace('\n', '')
                line = lines[i].split(' ')
                line[0] = str(int(float(line[0])))
                temp = line[2]
                line[2] = "{:6f}".format(1 - float(line[1]))
                line[1] = temp
                temp = line[4][:-1]
                line[4] = line[3]
                line[3] = temp
                lines[i] = ' '.join(line) + '\n'
            if __ * 90 == angle - 90:
                with open(save_txt, 'w') as t:
                    t.writelines(lines)


def adjust_brightness_contrast(image, brightness_factor=1.7, contrast_factor=2):
    """
    调整图片的亮度和对比度。

    :param image: 图片文件
    :param brightness_factor: 亮度调整因子，大于1为增加亮度，小于1为降低亮度
    :param contrast_factor: 对比度调整因子，大于1为增加对比度，小于1为降低对比度
    """

    # 调整亮度
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(brightness_factor)

    # 调整对比度
    enhancer = ImageEnhance.Contrast(bright_image)
    final_image = enhancer.enhance(contrast_factor)

    # 保存或显示调整后的图片
    # final_image.show()  # 显示图片
    # final_image.save('adjusted_image.jpg')  # 保存图片

    return final_image


def add_gaussian_noise(image, mean=0, std_dev=0.2):
    """
    给图像添加高斯噪声。

    参数:
        image (PIL.Image.Image): 输入图像。
        mean (float): 噪声的均值，默认为0。
        std_dev (float): 噪声的标准差，默认为0.5。

    返回:
        PIL.Image.Image: 添加了高斯噪声的图像。
    """
    # 将PIL图像转换为NumPy数组
    image_array = np.array(image)

    # 将图像数据类型转换为浮点数并归一化到[0, 1]区间
    image_array = image_array.astype(np.float32) / 255.0

    # 生成高斯噪声
    noise = np.random.normal(mean, std_dev, image_array.shape)

    # 添加噪声
    noisy_image_array = image_array + noise

    # 确保像素值在0到1之间
    noisy_image_array = np.clip(noisy_image_array, 0, 1)

    # 反归一化并将数据类型转换回uint8
    noisy_image_array = (noisy_image_array * 255).astype(np.uint8)

    # 将NumPy数组转换回PIL图像
    noisy_image = Image.fromarray(noisy_image_array)

    return noisy_image


def img_shrink(img_path, label_path, img_save_path, label_save_path):
    def corner_case(xi, yi, wide, high, width, height, width_add_length, height_add_length, n):
        x = xi * wide - (xi * width_add_length if xi != 0 else 0)
        y = yi * high - (yi * height_add_length if yi != 0 else 0)
        x1 = x + wide
        y1 = y + high
        x = max(x, 0)
        y = max(y, 0)
        x1 = min(x1, width)
        y1 = min(y1, height)
        return x, y, x1, y1

    def Img_Segmentation(img, n):
        img_List = []
        # 获取图像尺寸
        width, height = img.size[:2]
        n1 = width / 512
        n2 = height / 512
        n3 = max(n1, n2)
        if n <= n3:
            print("分割比例过小，请重新输入")
            return img_List
        height_base = 512
        width_base = 512
        # 计算由于重叠导致的额外长度
        height_add_length = (512 * n - height) / (n - 1)
        width_add_length = (512 * n - width) / (n - 1)
        # 遍历所有分割块，进行图像分割
        for i in range(1, n * n + 1):
            xi, yi = (i - 1) % n, (i - 1) // n

            # 处理边界情况
            if xi == 0 or xi == n - 1 or yi == 0 or yi == n - 1:
                x, y, x1, y1 = corner_case(xi, yi, width_base, height_base, width, height,
                                           width_add_length, height_add_length, n)
            else:
                # 计算非边界分割块的坐标
                x = width_base * xi - width_add_length * xi
                y = height_base * yi - height_add_length * yi
                x1 = x + width_base
                y1 = y + height_base
            # 进行图像分割，并添加到结果列表中
            img_seg = img.crop((x, y, x1, y1))
            image_seg = image(img_seg, x, y, x1, y1)
            img_List.append(image_seg)

        return img_List

    def read_label(label_path):
        """
        读取标签文件并将其转换为列表格式。

        参数:
        label_path: str - 标签文件的路径。

        返回值:
        list - 包含标签的列表，每个标签是一个整数列表。
        """
        label_list = []  # 初始化存储标签的列表

        # 打开并读取标签文件内容
        with open(label_path, 'r') as file:
            for line in file:
                # 将每行文本转换为整数列表
                number_list = [float(number) for number in line.split()]
                label_list.append(number_list)  # 将转换后的整数列表添加到标签列表中

        return label_list

    def init_label(label_list, width, height):

        # 遍历二维列表的每一行，并对元素进行处理
        for row_index, row in enumerate(label_list):
            # 计算并更新奇数索引位置的元素值
            for col_index in range(1, len(row), 2):
                label_list[row_index][col_index] *= width
            # 计算并更新偶数索引位置的元素值
            for col_index in range(2, len(row), 2):
                label_list[row_index][col_index] *= height
        return

    def update_label(img, label_list, width, height):
        for row in label_list:
            # 判断标签是否位于当前分割图像内
            if row[1] > img.x and row[1] < img.x1 and row[2] > img.y and row[2] < img.y1:
                # 计算标签在分割图像内的新位置，并更新到错误列表中
                x = row[1] - row[3] / 2
                y = row[2] - row[4] / 2
                x1 = row[1] + row[3] / 2
                y1 = row[2] + row[4] / 2
                x = max(x, img.x)
                y = max(y, img.y)
                x1 = min(x1, img.x1)
                y1 = min(y1, img.y1)
                new_centre_width = (x1 + x) / 2
                new_centre_height = (y1 + y) / 2
                update_label_list = [row[0], (new_centre_width - img.x) / width, (new_centre_height - img.y) / height,
                                     (x1 - x) / width, (y1 - y) / height]
                img.error_list.append(update_label_list)
        return

    #new_width, new_height = input("请输入收缩后图片尺寸：").split()
    new_width = 2048
    new_height = 2048
    x = int(max(new_width / 512, new_height / 512) + 1)
    img = Image.open(img_path)
    label_list = read_label(label_path)
    id = 0
    if img is not None:
        resized_img = img.resize((new_width, new_height))
        init_label(label_list, resized_img.width, resized_img.height)
        img_List = Img_Segmentation(resized_img, x)
        for idx, img_seg in enumerate(img_List):
            update_label(img_seg, label_list, img_seg.img.width, img_seg.img.height)
            if img_seg.error_list:
                for i in img_seg.error_list:
                    if i[1] > 1 or i[2] > 1 or i[3] > 1 or i[4] > 1:
                        print("Error: The label is out of range. Please check the label file.")
                    if i[1] - i[3] / 2 < -0.01 or i[2] - i[4] / 2 < -0.01 or i[1] + i[3] / 2 > 1.01 or i[2] + i[
                        4] / 2 > 1.01:
                        print("Error: The label is out of range. Please check the label file.")
                    # draw=ImageDraw.Draw(img_seg.img)
                    # draw.rectangle(((i[1]-i[3]/2)*img_seg.img.width,(i[2]-i[4]/2)*img_seg.img.height,(i[1]+i[3]/2)*img_seg.img.width,(i[2]+i[4]/2)*img_seg.img.height),outline='red',width=2)
                    img_seg.img.save(img_save_path[:-4] + f"_shrink{id}.bmp")
                    with open(label_save_path[:-4] + f"_shrink{id}.txt", 'w') as f:
                        for i in img_seg.error_list:
                            for j in i:
                                f.write(str(j))
                                f.write(' ')
                            f.write("\n")
                        f.close()
                    id += 1
    else:
        print("Error: Image not found. Please check the file path.")


def img_enlarge(img_path, label_path, img_save_path, label_save_path):
    def corner_case(xi, yi, wide, high, width, height, width_add_length, height_add_length, n):
        x = xi * wide - (xi * width_add_length if xi != 0 else 0)
        y = yi * high - (yi * height_add_length if yi != 0 else 0)
        x1 = x + wide
        y1 = y + high
        x = max(x, 0)
        y = max(y, 0)
        x1 = min(x1, width)
        y1 = min(y1, height)
        return x, y, x1, y1

    def Img_Segmentation(img, n):
        img_List = []
        # 获取图像尺寸
        width, height = img.size[:2]
        n1 = width / 512
        n2 = height / 512
        n3 = max(n1, n2)
        if n <= n3:
            print("分割比例过小，请重新输入")
            return img_List
        height_base = 512
        width_base = 512
        # 计算由于重叠导致的额外长度
        height_add_length = (512 * n - height) / (n - 1)
        width_add_length = (512 * n - width) / (n - 1)
        # 遍历所有分割块，进行图像分割
        for i in range(1, n * n + 1):
            xi, yi = (i - 1) % n, (i - 1) // n

            # 处理边界情况
            if xi == 0 or xi == n - 1 or yi == 0 or yi == n - 1:
                x, y, x1, y1 = corner_case(xi, yi, width_base, height_base, width, height,
                                           width_add_length, height_add_length, n)
            else:
                # 计算非边界分割块的坐标
                x = width_base * xi - width_add_length * xi
                y = height_base * yi - height_add_length * yi
                x1 = x + width_base
                y1 = y + height_base
            # 进行图像分割，并添加到结果列表中
            img_seg = img.crop((x, y, x1, y1))
            image_seg = image(img_seg, x, y, x1, y1)
            img_List.append(image_seg)

        return img_List

    def read_label(label_path):
        """
        读取标签文件并将其转换为列表格式。

        参数:
        label_path: str - 标签文件的路径。

        返回值:
        list - 包含标签的列表，每个标签是一个整数列表。
        """
        label_list = []  # 初始化存储标签的列表

        # 打开并读取标签文件内容
        with open(label_path, 'r') as file:
            for line in file:
                # 将每行文本转换为整数列表
                number_list = [float(number) for number in line.split()]
                label_list.append(number_list)  # 将转换后的整数列表添加到标签列表中

        return label_list

    def init_label(label_list, width, height):

        # 遍历二维列表的每一行，并对元素进行处理
        for row_index, row in enumerate(label_list):
            # 计算并更新奇数索引位置的元素值
            for col_index in range(1, len(row), 2):
                label_list[row_index][col_index] *= width
            # 计算并更新偶数索引位置的元素值
            for col_index in range(2, len(row), 2):
                label_list[row_index][col_index] *= height
        return

    def update_label(img, label_list, width, height):
        for row in label_list:
            # 判断标签是否位于当前分割图像内
            if row[1] > img.x and row[1] < img.x1 and row[2] > img.y and row[2] < img.y1:
                # 计算标签在分割图像内的新位置，并更新到错误列表中
                x = row[1] - row[3] / 2
                y = row[2] - row[4] / 2
                x1 = row[1] + row[3] / 2
                y1 = row[2] + row[4] / 2
                x = max(x, img.x)
                y = max(y, img.y)
                x1 = min(x1, img.x1)
                y1 = min(y1, img.y1)
                new_centre_width = (x1 + x) / 2
                new_centre_height = (y1 + y) / 2
                update_label_list = [row[0], (new_centre_width - img.x) / width, (new_centre_height - img.y) / height,
                                     (x1 - x) / width, (y1 - y) / height]
                img.error_list.append(update_label_list)
        return

    def expand_update_label(img_seg,n,expand_width,expand_height):
        if img_seg.error_list:
            for row in img_seg.error_list:
                x_centre=row[1]*512*n+expand_width
                y_centre=row[2]*512*n+expand_height
                row[3]=row[3]*n
                row[4]=row[4]*n
                row[1]=x_centre/img_seg.img.width
                row[2]=y_centre/img_seg.img.height
        return

    n=2
    if(n<1) :
        print("放大比例不合法，请重新输入")
        return
    id=0
    img = Image.open(img_path)
    label_list = read_label(label_path)
    if img is not None:
        new_width=int(img.width*n)
        new_height=int(img.height*n)
        resized_img=img.resize((new_width,new_height))
        # else:
        #     resized_img=img
        init_label(label_list, resized_img.width, resized_img.height)
        x=int(max(resized_img.width,resized_img.height)/512+1)
        img_List=Img_Segmentation(resized_img,x)
        for idx,img_seg in enumerate(img_List):
            update_label(img_seg, label_list,img_seg.img.width,img_seg.img.height)
            # if n<1:
            #     new_width=int(img.width*n)
            #     new_height=int(img.height*n)
            #     img_seg.img=img_seg.resize((new_width,new_height))
            #     expand_width=int((512-img_seg.img.width)/2)
            #     expand_height=int((512-img_seg.img.height)/2)
            #     img_seg.img=ImageOps.expand(img_seg.img,(expand_width,expand_height,expand_width,expand_height),fill=(255,255,255))
            #     expand_update_label(img_seg,n,expand_width,expand_height)
            if img_seg.error_list:
                for i in img_seg.error_list:
                    if i[1]>1 or i[2]>1 or i[3]>1 or i[4]>1:
                        print("Error: The label is out of range. Please check the label file.")
                    if i[1]-i[3]/2<-0.01 or i[2]-i[4]/2<-0.01 or i[1]+i[3]/2>1.01 or i[2]+i[4]/2>1.01:
                        print("Error: The label is out of range. Please check the label file.")
                    #draw=ImageDraw.Draw(img_seg.img)
                    #draw.rectangle(((i[1]-i[3]/2)*img_seg.img.width,(i[2]-i[4]/2)*img_seg.img.height,(i[1]+i[3]/2)*img_seg.img.width,(i[2]+i[4]/2)*img_seg.img.height),outline='red',width=2)
                img_seg.img.save(img_save_path[:-4]+f"_big{id}.bmp")
                with open(label_save_path[:-4]+f"_big{id}.txt",'w') as f:
                    for i in img_seg.error_list:
                        for j in i:
                            f.write(str(j))
                            f.write(' ')
                        f.write("\n")
                    f.close()
                id+=1
    else:
        print("Error: Image not found. Please check the file path.")


def img_narrow(img_path, label_path, img_save_path, label_save_path):
    def corner_case(xi, yi, wide, high, width, height, width_add_length, height_add_length, n):
        x = xi * wide - (xi * width_add_length if xi != 0 else 0)
        y = yi * high - (yi * height_add_length if yi != 0 else 0)
        x1 = x + wide
        y1 = y + high
        x = max(x, 0)
        y = max(y, 0)
        x1 = min(x1, width)
        y1 = min(y1, height)
        return x, y, x1, y1

    def Img_Segmentation(img, n):
        img_List = []
        # 获取图像尺寸
        width, height = img.size[:2]
        n1 = width / 512
        n2 = height / 512
        n3 = max(n1, n2)
        if n <= n3:
            print("分割比例过小，请重新输入")
            return img_List
        height_base = 512
        width_base = 512
        # 计算由于重叠导致的额外长度
        height_add_length = (512 * n - height) / (n - 1)
        width_add_length = (512 * n - width) / (n - 1)
        # 遍历所有分割块，进行图像分割
        for i in range(1, n * n + 1):
            xi, yi = (i - 1) % n, (i - 1) // n

            # 处理边界情况
            if xi == 0 or xi == n - 1 or yi == 0 or yi == n - 1:
                x, y, x1, y1 = corner_case(xi, yi, width_base, height_base, width, height,
                                           width_add_length, height_add_length, n)
            else:
                # 计算非边界分割块的坐标
                x = width_base * xi - width_add_length * xi
                y = height_base * yi - height_add_length * yi
                x1 = x + width_base
                y1 = y + height_base
            # 进行图像分割，并添加到结果列表中
            img_seg = img.crop((x, y, x1, y1))
            image_seg = image(img_seg, x, y, x1, y1)
            img_List.append(image_seg)

        return img_List

    def read_label(label_path):
        """
        读取标签文件并将其转换为列表格式。

        参数:
        label_path: str - 标签文件的路径。

        返回值:
        list - 包含标签的列表，每个标签是一个整数列表。
        """
        label_list = []  # 初始化存储标签的列表

        # 打开并读取标签文件内容
        with open(label_path, 'r') as file:
            for line in file:
                # 将每行文本转换为整数列表
                number_list = [float(number) for number in line.split()]
                label_list.append(number_list)  # 将转换后的整数列表添加到标签列表中

        return label_list

    def init_label(label_list, width, height):

        # 遍历二维列表的每一行，并对元素进行处理
        for row_index, row in enumerate(label_list):
            # 计算并更新奇数索引位置的元素值
            for col_index in range(1, len(row), 2):
                label_list[row_index][col_index] *= width
            # 计算并更新偶数索引位置的元素值
            for col_index in range(2, len(row), 2):
                label_list[row_index][col_index] *= height
        return

    def update_label(img, label_list, width, height):
        for row in label_list:
            # 判断标签是否位于当前分割图像内
            if row[1] > img.x and row[1] < img.x1 and row[2] > img.y and row[2] < img.y1:
                # 计算标签在分割图像内的新位置，并更新到错误列表中
                x = row[1] - row[3] / 2
                y = row[2] - row[4] / 2
                x1 = row[1] + row[3] / 2
                y1 = row[2] + row[4] / 2
                x = max(x, img.x)
                y = max(y, img.y)
                x1 = min(x1, img.x1)
                y1 = min(y1, img.y1)
                new_centre_width = (x1 + x) / 2
                new_centre_height = (y1 + y) / 2
                update_label_list = [row[0], (new_centre_width - img.x) / width, (new_centre_height - img.y) / height,
                                     (x1 - x) / width, (y1 - y) / height]
                img.error_list.append(update_label_list)
        return

    def expand_update_label(img_seg,n,expand_width,expand_height):
        if img_seg.error_list:
            for row in img_seg.error_list:
                x_centre=row[1]*512*n+expand_width
                y_centre=row[2]*512*n+expand_height
                row[3]=row[3]*n
                row[4]=row[4]*n
                row[1]=x_centre/img_seg.img.width
                row[2]=y_centre/img_seg.img.height
        return

    n=0.5
    if(n>1) :
        print("缩小比例不合法，请重新输入")
        return
    id=0
    img = Image.open(img_path)
    label_list = read_label(label_path)
    if img is not None:
        resized_img=img
        init_label(label_list, resized_img.width, resized_img.height)
        x=int(max(resized_img.width,resized_img.height)/512+1)
        img_List=Img_Segmentation(resized_img,x)
        for idx,img_seg in enumerate(img_List):
            update_label(img_seg, label_list,img_seg.img.width,img_seg.img.height)
            new_width=int(img_seg.img.width*n)
            new_height=int(img_seg.img.height*n)
            img_seg.img=img_seg.img.resize((new_width,new_height))
            expand_width=int((512-img_seg.img.width)/2)
            expand_height=int((512-img_seg.img.height)/2)
            img_seg.img=ImageOps.expand(img_seg.img,(expand_width,expand_height,expand_width,expand_height),fill=(255,255,255))
            expand_update_label(img_seg,n,expand_width,expand_height)
            if img_seg.error_list:
                for i in img_seg.error_list:
                    if i[1]>1 or i[2]>1 or i[3]>1 or i[4]>1:
                        print("Error: The label is out of range. Please check the label file.")
                    if i[1]-i[3]/2<-0.01 or i[2]-i[4]/2<-0.01 or i[1]+i[3]/2>1.01 or i[2]+i[4]/2>1.01:
                        print("Error: The label is out of range. Please check the label file.")
                    #draw=ImageDraw.Draw(img_seg.img)
                    #draw.rectangle(((i[1]-i[3]/2)*img_seg.img.width,(i[2]-i[4]/2)*img_seg.img.height,(i[1]+i[3]/2)*img_seg.img.width,(i[2]+i[4]/2)*img_seg.img.height),outline='red',width=2)
                img_seg.img.save(img_save_path[:-4]+f"_small{id}.bmp")
                with open(label_save_path[:-4]+f"_small{id}.txt",'w') as f:
                    for i in img_seg.error_list:
                        for j in i:
                            f.write(str(j))
                            f.write(' ')
                        f.write("\n")
                    f.close()
                id+=1
    else:
        print("Error: Image not found. Please check the file path.")
