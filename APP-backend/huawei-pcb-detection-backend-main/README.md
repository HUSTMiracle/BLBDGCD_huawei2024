# 华为挑战杯 鸿蒙APP后端 拔萝卜的工程队

这里是鸿蒙APP的后端，默认使用5001端口处理http请求，5002端口处理websocket请求

运行前请按照`dotenv-example`内的信息将环境变量配置好，以便后端调用modelarts以及IoT平台的服务

## 整体架构

```
.
├── data # postgres/opengauss的持久化存储目录
├── docker-compose-dev.yml #没啥用了，只起了一个数据库
├── docker-compose.yml #使用Dockerfile-compose， 并挂载若干本地目录到容器中
├── Dockerfile-compose #是Dockerfile， 只复制requirements.txt并安装环境依赖，源码直接挂载本地卷
├── Dockerfile-full    #是Dockerfile, 将所有python源码打包成一个Docker镜像
├── .env #存储环境变量作为配置
├── README.md
├── requirements.txt
├── resources # 存放模型的资源文件夹
│   └── 600ep.onnx
├── src
│   ├── apig_sdk # 华为提供的api gateway, 用来接收AccessKey和SecretKey并为请求签名（鉴权用）
│   ├── backend_model  # 数据库相关，支持postgrasql/opengauss
│   ├── controller # 标准的controller函数
│   ├── main
│   ├── router  # 使用了Flask的BluePrint包来注册路由
│   ├── util 
|       ├── draw_boxes.py # 对上传推理完毕的图片绘制矩形框
|       ├── get_token.py # modelarts鉴权相关，获取token
|       ├── inference_ak_sk.py # 使用华为的modelarts模型进行推理，使用ak_sk模式鉴权
|       ├── inference_local_v9.py # 加载本地的yolov9模型推理
|       ├── inference_token.py # 使用华为的modelarts模型进行推理，使用token模式鉴权
|       ├── iot.py # 从IoT平台获取数据的util
|       └── yolo_v9 # 模型组同学的成果，用来加载本地模型
└── uploads # 上传的文件会被持久化存储在这里，其对应的名称则会被存储到数据库中
    ├── xxx.jpg
```


## How to Run

```bash
sudo docker-compose -f docker-compose.yml --compatibility up
```

## Refs
[modelarts api](https://support.huaweicloud.com/intl/en-us/inference-modelarts/inference-modelarts-0018.html)
