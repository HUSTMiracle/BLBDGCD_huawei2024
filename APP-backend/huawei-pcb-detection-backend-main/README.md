# 华为挑战杯 鸿蒙APP后端 拔萝卜的工程队

这里是鸿蒙APP的后端，默认使用5001端口处理http请求，5002端口处理websocket请求

运行前请按照`dotenv-example`内的信息将环境变量配置好，以便后端调用modelarts以及IoT平台的服务


## How to Run

```bash
sudo docker-compose -f docker-compose.yml --compatibility up
```

## Refs
[modelarts api](https://support.huaweicloud.com/intl/en-us/inference-modelarts/inference-modelarts-0018.html)
