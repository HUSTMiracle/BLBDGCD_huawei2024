import {Core, ExpectExtend,InstrumentLog} from 'deccjsunit/index'

export default {
    data: {
        title: ""
    },
    onInit() {
        this.title = this.$t('strings.world');
    },
    onShow() {
        console.info('onShow finish')
        const core = Core.getInstance()
        const expectExtend = new ExpectExtend({
            'id': 'extend'
        })
        const instrumentLog = new InstrumentLog({
            'id': 'report', 'unity': 'true'
        })
        core.addService('expect', expectExtend)
        core.addService('report', instrumentLog)
        core.init()
        core.subscribeEvent('spec', instrumentLog)
        core.subscribeEvent('suite', instrumentLog)
        core.subscribeEvent('task', instrumentLog)
        const configService = core.getDefaultService('config')
        configService.setConfig(this)
        require('../../../test/List.test')
        core.execute()
    },
    onReady() {
    },
}