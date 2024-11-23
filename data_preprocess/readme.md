# 拔萝卜的工程队 数据处理代码  
运行命令：
```bash
python preprocess.py --folder path/to/your/image_folder --txt_folder path/to/your/txt_folder \
--image_output_folder path/to/your/image_out_folder --txt_output_folder path/to/your/txt_out_folder
```
训练集图片需要为bmp格式，标签需已经转化为标准格式（即和赛题提供的样例一致），default mode=every  
我们在训练过程中对不同的训练集采用了不同的训练模式，可以通过--mode参数来选择，具体如下：  
1. 对于初赛、复赛样例集，mode=every  
2. 对于外源数据集和自制数据集，mode=random  
3. 对于初赛样例集，再进行一次单独处理，mode=right  
我们将以上三步获得的训练集合并在一起，构成我们最终的数据集。