# HarmonyOS质检app

## 功能介绍

质检app运行环境为HarmonyOS 4.0.0（真机），api版本是6，采用JS FA模型开发，可针对Mouse_bite、Open_circuit、Short、Spur、Spurious_copper等瑕疵类型的pcb板进行瑕疵检测。整个 app 共分为历史速览、异常检测、历史记录三大模块

### 异常检测

质检员可以自由的对有瑕疵的pcb板进行拍摄，调用云端modelarts推理，并在首页返回推理结果图（包含该pcb板的瑕疵位置、瑕疵类型、检测置信度）

### 历史记录

质检员可以根据检测日期和图片名称（以时间戳命名）精准定位之前检测过的pcb板，每条历史记录包含该板的推理结果图，同时以饼状图的形式统计该板不同瑕疵类型占比和异常总数

### 历史速览

为方便质检员快速查询，该模块记录了历史最新5次的检测结果，以滑动窗口和缩略图的形式呈现，缩略图支持点击查看大图功能

## 程序概要设计

### 前端

#### 项目结构

```SHELL
pages
├───index
 	 index.css
	index.hml
	index.js
├───page
        page.css
        page.hml
        page.js
│───splash
        page.css
        page.hml
        page.js
```

前端由四个页面组成，分别是：启动页、异常检测、首页、历史记录，通过`@ohos.router`的router相互路由，网络请求均通过`@ohos.net.http`原生http进行promise异步请求

#### 异常检测

异常检测调用camera组件，通过其takePhoto中的complete方法返回的`result.uri`通过简单的正则拿到照片沙箱路径，创建一个原始二进制缓冲区`ArrayBuffer`，通过`@ohos.fileio`的fileIO实现图片文件同步读写，最后将读出的字节序列字符串放入请求体中传给后端

#### 首页

进度条

设立一个`isLoading`的flag，当质检员拍照，flag置为true，调用`progressLoading`启动进度条，其在 flag 为 true 时递增 `progress`组件中的`percent`，并使用 `setTimeout` 继续调用自己，实现循环更新，当`modelarts`推理返回推理结果图时flag置为false，此时首页渲染最新的检测结果图

滑动窗口

使用 `swiper` 组件实现图片的分页滑动显示，每个页面最多显示三张图片。通过 `recentImages` 数组和 `slice` 方法，动态地从数组中提取图片进行显示。

#### 历史记录

在生命周期`onInit()`方法中，使用全局变量 `globalThis.value` 来获取图片 ID，构建图片 URL，实现image标签src请求图片

维护一个`detectionClasses` 数组来存储每个异常类别的**名称**和**占比**，后端返回的推理样例如下：

```json
{
  "filename": "test.png",
  "result": {
    "detection_boxes": [
      [
        1580.695556640625,
        1063.1583251953125,
        1632.359619140625,
        1074.5572509765625
      ],
      [
        1577.9443359375,
        1063.1236572265625,
        1629.8369140625,
        1074.6903076171875
      ],
      [
        707.0120239257812,
        517.9932250976562,
        747.9051513671875,
        527.4000854492188
      ]
    ],
    "detection_classes": [
      "Spurious_copper",
      "Spurious_copper",
      "Spurious_copper"
    ],
    "detection_scores": [
      0.2881912589073181,
      0.24983583390712738,
      0.19878366589546204
    ]
  },
  "source": "ModelArts",
  "success": true
}
```

利用HarmonyOS内置的原子服务，可实现三次敲击屏幕放大，双指拖动图片，以查看检测结果图详情
