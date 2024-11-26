# huawei-cloud-orangepi-controller
OrangePI5_PLUS 开发板上的controller


### 环境配置
```
sudo apt-get install pkg-config libcairo2-dev

conda activate controlconda create --name controller python=3.11ler
pip install -r requirements.txt
```

### 运行

```
python3 ./src/file_server.py

python3 ./src/main.py
```

### 相机使用
本项目使用海康威视安防相机，开发使用海康官方SDK,本项目为SDK进行了python封装，方便兼容。


#### Usage
切换到 /src/HCNetPythonSDK-Linux 目录下
```
python setup.py --install 
```
即可将SDK安装到conda环境中

#### Config
海康相机的使用需要使开发板和摄像机在同一网段下，可以使用SADP工具设置海康相机的ip
连接相机的网口需要按照上述要求配置静态IP,这里设置为202.114.213.56，端口号8000

拍照获得的图像保存在 /asset/ 路径下

相机推流使用海康默认rtsp地址。

海康威视所需SDK需要到官网下载，下载arm版本的SDK之后需要把所有文件(包括xml和so)复制到hklib目录下
#### Example
```
from hikvision.hikvision import HIKVisionSDK

LIB_DIR = '/home/root/PythonProject/HCNetPythonSDK-Linux/hklib'

sdk = HIKVisionSDK(lib_dir=LIB_DIR,
                   username='admin',
                   ip='192.168.1.124',
                   password='Admin12345')
try:
    sdk.init()
except Exception as e:
    print(e)
    print('Errcode ', sdk.err_code)
else:
    ok = sdk.take_picture('/tmp/jjjj.jpg', release_resources=False)
    print('ok1', ok)
    ok = sdk.take_picture('/tmp/jjjj3.jpg', release_resources=False)
    print('ok2', ok)
    value = sdk.get_zoom(release_resources=False)
    print('zoom value', value)
    ok = sdk.set_zoom(zoom=10, release_resources=True)
    print('zoom ok', ok)


print(sdk.get_infrared_value())
```
### 模型使用

提供了yolov9c,yolov9s,yolov9e的onnx模型权重，开发者可以自行更改__init__.py里controller使用的模型。

### 工业相机

将安防相机替换为清晰度更高，分辨率更高，对焦更灵活的工业相机，使用海康威视机器人，MVS SDK可以在官网上下载
，只需在 videohandle/convertpixel.py 中更改主要文件的挂载路径即可。

相机的参数可以通过 videohandle/FeatureFile.ini 实现简单更改，具体参数请见海康手册。