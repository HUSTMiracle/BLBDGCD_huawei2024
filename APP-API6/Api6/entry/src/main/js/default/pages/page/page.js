import router from '@system.router';
import http from '@ohos.net.http';

export default {
    data: {
        imageId: "",
        imageUrl: "",
        getOneImageHistory: "http://10.12.168.52:5001/api/v1/get_one_history",
        detail_id: "",
        detectionClasses: {},
        totalCount: 0,
        theShow: "",
    },
    onInit() {
        let before = "http://10.12.168.52:5001/images/";
        this.imageId = globalThis.value && globalThis.value.data;
        this.imageUrl = before + this.imageId;
    },
    onShow() {
        this.getOneHistory();
    },
    goBack() {
        router.back();
    },
    switch() {
        if(this.theShow === "show")this.theShow="";
        else this.theShow="show";
    },
    getOneHistory() {
        const testUrl = this.getOneImageHistory + '?image_id=' + encodeURIComponent(this.imageId);
        let httpRequest = http.createHttp();
        let promise = httpRequest.request(
            testUrl,
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
            console.info('code:' + data.responseCode);

            const parsedResult = JSON.parse(data.result);
            this.detail_id = parsedResult.result.image_id;
            // 获取classes
            const detectionClasses = parsedResult.result.inference_result.result.detection_classes;
            this.totalCount = detectionClasses.length;

            // 统计每个类别的出现次数
            const classCount = {};
            detectionClasses.forEach(className => {
                if (classCount[className]) {
                    classCount[className]++;
                } else {
                    classCount[className] = 1;
                }
            });

            // chartDataset format
            this.detectionClasses = Object.entries(classCount).map(([name, count]) => ({
                name,
                value: Math.round((count / this.totalCount) * 100)
            }));
        }).catch((err) => {
            console.info('error:' + JSON.stringify(err));
        })
    },
}

