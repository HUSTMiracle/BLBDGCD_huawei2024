import router from '@system.router';
import featureAbility from '@ohos.ability.featureAbility';

export default {
    data: {
        // 声明数据
    },
    onInit() {
        // 页面初始化时触发
    },
    onReady() {
        setTimeout(() => {
            this.navigateToMainPage();
        }, 2000);
    },
    navigateToMainPage() {
        router.replace({
            uri: 'pages/index/index',
        });
    },
    onBackPress() {
        return true; // 返回 true 表示消费了返回键事件，防止退出应用
    }
}
