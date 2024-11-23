import prompt from '@system.prompt';
import http from '@ohos.net.http';
import fileIO from '@ohos.fileio';
import router from '@system.router';
import webSocket from '@ohos.net.webSocket';

export default {
    data: {
        CurrentPage: "Page0",
        theColor: "#ffffff",
        MyBlue: "#44cef6",
        src: "",
        lastImage: "",
        urls: {
            imageDetection: "http://10.12.168.10:5001/api/v1/inference_binary",
            getAllHistories: "http://10.12.168.10:5001/api/v1/get_all_histories",
            getLastImage: "http://10.12.168.10:5001/images/",
            camera: 'ws://10.12.168.10:5002/66b8276ed8c5af5f58f200d3_PCB_ON_BELT'
        },
        pages: {
            Page0: { color: "SelectedColor", icon: ["hidden", ""] },
            Page1: { color: "", icon: ["", "hidden"] },
            Page2: { color: "", icon: ["", "hidden"] }
        },
        historyList: [],
        page : 1,
        limit : 50,
        date : 20240824,
        imgNum: 1,
        percent: 0,
        timer: null,
        isLoading: false,
        dateValue: '2024-08-25',
        dialogShow: 0,
        imageUrl0: ''
    },

    // 进度条加载进度循环控制
    onInit() {

        this.startLoading();



    },
    onShow() {
        console.log("123");
        this.theTimer();
    },
    theTimer(){
        console.log("1515151515115");
        this.flawsDetection();

    },
    startLoading() {
        console.log("234");
        const incrementProgress = () => {
            this.percent = (this.percent + 1) % 101; // 增加百分比，到100后重置为0
            this.timer = setTimeout(incrementProgress, 50); // 每50毫秒更新一次
        };
        incrementProgress();
    },
    onDestroy() {
        if (this.timer) {
            clearTimeout(this.timer);
            this.timer = null;
        }
    },

    // 切换页面逻辑
    switchButtonColor(PageId) {
        const page = this.pages[PageId];
        page.color = page.color ? "" : "SelectedColor";
        page.icon = page.icon.reverse();
    },
    switchLastIcon(PageId) {
        const page = this.pages[PageId];
        page.icon = ["", "hidden"];
    },
    switchPages(PageId) {
        if (PageId === this.CurrentPage) {
            console.log("Choose The Same Page: " + PageId);
            return;
        }
        this.switchButtonColor(this.CurrentPage);
        this.switchLastIcon(this.CurrentPage);
        this.switchButtonColor(PageId);
        this.CurrentPage = PageId;
    },

    // 文件读写
    readImageFile(imagePath) {
        const file  = fileIO.openSync(imagePath, 0o2);
        const stat = fileIO.statSync(imagePath);
        let buffer = new ArrayBuffer(stat.size);
        fileIO.readSync(file, buffer);
        fileIO.closeSync(file);
        return buffer;
    },
    uint8ArrayToString(buffer) {
        let uint8Array = new Uint8Array(buffer);
        return uint8Array.join(',');
    },

    // 进度条显示隐藏
    progressStartLoading() {
        this.percent = 0;
        this.progressLoading();
    },
    progressStopLoading() {
        this.isLoading = false;
        this.percent = 0;
    },
    progressLoading() {
        if (this.isLoading) {
            this.percent = (this.percent + 1) % 101;
            setTimeout(() => this.progressLoading(), 50);
        }
    },

    // 摄像头
    cameraError() {
        prompt.showToast({
            message: "授权失败！"
        })
    },
    takePhoto() {
        const _this = this;
        console.log('------------------------------>takePhoto');
        let camera = this.$element('camera');

        camera.takePhoto({
            quality: 'high',
            success(result) {
                _this.src = result.uri
                // 加载进度条
                _this.isLoading = true;
                _this.progressStartLoading();
                console.log("拍照成功路径==>>" + _this.src);
            },
            fail(result) {
                console.info('-------------fail------' + result);
            },
            complete(result) {
                console.info('-------------complete------' + result)

                const deleteSrc = "file://";
                const appSrc = "file:///data/data/com.example.api6/files/";

                const { src } = _this;  // 确保使用相同的 src
                // imagePath
                const [, imagePath] = src.match(`^${deleteSrc}(.*)$`) || [];

                // path_homepage
                const [, relativePath] = src.match(`^${appSrc}(.*)$`) || [];
                const path_homepage = relativePath ? `internal://app/${relativePath}` : '';

                console.info('-------------success------' + path_homepage);

                const fileName = imagePath.split('/').pop();
                const urlWithQuery = _this.urls.imageDetection + '?filename=' + encodeURIComponent(fileName);
                let httpRequest = http.createHttp();
                let promise = httpRequest.request(
                    urlWithQuery,
                    {
                        method: http.RequestMethod.POST,
                        header: {
                            'Content-Type': 'text/plain',
                        },
                        extraData: _this.uint8ArrayToString(_this.readImageFile(imagePath)),
                        readTimeout: 60000,
                        connectTimeout: 60000
                    },
                );
                promise.then((data) => {
                    console.info('Result:' + data.result);
                    console.info('code:' + data.responseCode);
                    const parseRes = JSON.parse(data.result);
                    const lastImageID = parseRes.filename;
                    _this.lastImage = _this.urls.getLastImage + lastImageID;
                    // 加载完成，隐藏进度条
                    _this.isLoading = false;
                    _this.progressStopLoading();
                }).catch((err) => {
                    console.info('error:' + JSON.stringify(err));
                });
            }
        });
    },
    change(e){
        console.log(e.value);
        this.date = e.value;
    },

    // 获取所有拍照上传历史记录
    getAllHistories(e,num) {
        console.log("here");

        const urlWithQuery = `${this.urls.getAllHistories}?page=${encodeURIComponent(this.page)}&limit=${encodeURIComponent(this.limit)}&date=${encodeURIComponent(this.date)}`;
        let httpRequest = http.createHttp();
        let promise = httpRequest.request(
            urlWithQuery,
            {
                method: http.RequestMethod.GET,
                header: {
                    'Content-Type': 'application/json'
                },
                readTimeout: 30000,
                connectTimeout: 30000
            },
        );
        promise.then((data) => {
            console.info('Result:' + data.result);

            const parsedResult = JSON.parse(data.result);
            while(this.historyList.length >0)
                this.historyList.pop();
            this.imgNum = 1;
            parsedResult.result.map(item => {
                this.historyList.push(
                    {
                        number: this.imgNum,
                        imageName: item.image_id,
                        errorNum: item.detected_flaws
                    }
                )
                console.log("yes");
                this.imgNum++;
            })

            console.info('code:' + data.responseCode);
            console.info('header:' + JSON.stringify(data.header));
        }).catch((err) => {
            console.info('error:' + JSON.stringify(err));
        });
    },
    dateOnChange(e) {
        const month = String(e.month + 1).padStart(2, '0');
        const day = String(e.day).padStart(2, '0');

        this.dateValue = `${e.year}-${month}-${day}`;

        this.date = parseInt(this.dateValue.replace(/-/g, ''), 10);
    },
    // 每一条历史记录的具体信息
    imageDetail(imageName) {
        globalThis.value = {
            data: imageName,
        }
        router.push({
            uri: 'pages/page/page',
        });
    },

    // IOT flaws_detection
    flawsDetection() {
        const address = this.urls.camera;
        let ws = webSocket.createWebSocket();
        ws.connect(address, (err, value) => {
            (!err) ? console.log("Connected successfully") :  console.log("Connection failed. Err:" + JSON.stringify(err));
        });

        ws.on('message', (err, value) => {
            const res = JSON.parse(value);
            if (res.total_flaws_detected != this.dialogShow) {
                this.$element('dialogId').show();
                this.dialogShow = res.total_flaws_detected;
            }
            this.imageUrl0 = "";
            this.imageUrl0 = res.streaming_url;
        })

        ws.on('close', (err, value) => {
            console.log("on close, code is " + value.code + ", reason is " + value.reason);
        });
    }
}
