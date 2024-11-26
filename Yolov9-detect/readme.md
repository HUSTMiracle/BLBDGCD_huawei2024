# 华为云平台代码包
本文件夹用于在华为云modelarts上进行打榜。  
由于大小限制，我们无法将模型参数文件上传。你可以自行训练出best.onnx，放到本文件夹根目录下实现云侧自动判分。本模型包在custom_service.py里将华为默认的best.pt换成了best.onnx，如果你需要上传.pt文件，可以在custom_service.py中修改相应代码。  
模型包的具体结构详询华为官方文档。
