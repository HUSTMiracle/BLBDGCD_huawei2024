package com.pcbdetection.app;

import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Enable immersive mode
        Window window = getWindow();
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        window.setStatusBarColor(Color.TRANSPARENT);
        window.setNavigationBarColor(Color.TRANSPARENT);

        View decorView = window.getDecorView();
        decorView.setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE |
                        View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION |
                        View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN |
                        View.SYSTEM_UI_FLAG_HIDE_NAVIGATION |
                        View.SYSTEM_UI_FLAG_FULLSCREEN |
                        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        );

        // Adjust layout to avoid status bar and navigation bar obstruction
//        decorView.setOnApplyWindowInsetsListener((v, insets) -> {
//            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
//                int topInset = insets.getSystemWindowInsetTop();
//                int bottomInset = insets.getSystemWindowInsetBottom();
//                int leftInset = insets.getSystemWindowInsetLeft();
//                int rightInset = insets.getSystemWindowInsetRight();
//                v.setPadding(leftInset, topInset, rightInset, bottomInset);
//            } else {
//                v.setPadding(0, 0, 0, 0); // Fallback for older API levels
//            }
//            return insets;
//        });
    }
}
