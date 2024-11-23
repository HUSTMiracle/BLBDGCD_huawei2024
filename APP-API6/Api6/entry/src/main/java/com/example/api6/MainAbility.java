package com.example.api6;
import ohos.agp.window.service.WindowManager;
import ohos.ace.ability.AceAbility;
import ohos.aafwk.content.Intent;

public class MainAbility extends AceAbility {
    @Override
    public void onStart(Intent intent) {
        //获得当前窗口对象，添加app全屏显示（隐藏状态栏）
        super.getWindow().addFlags(WindowManager.LayoutConfig.MARK_FULL_SCREEN);
        super.onStart(intent);
    }

    @Override
    public void onStop() {
        super.onStop();
    }
}
