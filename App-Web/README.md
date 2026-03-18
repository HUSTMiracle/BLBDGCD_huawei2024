# App-Web

## 技术框架

- 应用形态：Web APP应用，使用Capacitor 容器进行Android移动端封装。
- 前端层：基于 Vue 3 进行组件化开发，页面由 `views` 组织，可复用组件实现在 `components`。
- 路由层：使用前端路由进行页面切换与导航管理，支持首页、历史记录、实时监控、设置等页面流转。
- 状态层：集中式状态管理用于保存检测结果、通知信息、主题配置等全局状态。
- 服务层： `services` 统一封装接口调用与业务请求，减少页面和后端协议的直接耦合。
- 构建层：使用 Vite 进行开发调试与生产打包，TypeScript 提供类型约束和更稳定的工程维护能力。

### APP 功能

- 图片采集与上传：支持从设备端采集或上传图片进行检测。
- 缺陷检测结果展示：对检测输出进行结构化展示，便于快速查看异常信息。
- 历史记录与详情：提供历史检测记录浏览与单条结果详情查看。
- 实时监控：支持实时检测场景下的结果查看与状态反馈。
- 主题与基础设置：提供主题切换和基础配置。

## 项目架构

```text
App-Web/
├─ src/
│  ├─ components/        # 通用组件
│  ├─ views/             # 页面视图
│  ├─ router/            # 路由配置
│  ├─ stores/            # 状态管理
│  ├─ services/          # 接口与业务服务
│  └─ config/            # 运行配置
├─ public/               # 静态资源
├─ android/              # Capacitor Android 工程
├─ package.json          # Node 依赖与脚本
├─ vite.config.ts        # Vite 配置
└─ capacitor.config.ts   # Capacitor 配置
```

## 构建 & 运行

- 安装依赖：`npm install`
- 本地开发：`npm run dev`
- 生产构建：`npm run build`
- 首次添加 Android 平台：`npx cap add android`
- 同步 Android 平台：`npx cap sync android`
- 拷贝前端资源到 Android：`npx cap copy android`
- 使用 Android Studio 打开 Android 工程：`npx cap open android`

## 原仓库链接

- https://github.com/zkf516/PCB-Flaws-Detection
