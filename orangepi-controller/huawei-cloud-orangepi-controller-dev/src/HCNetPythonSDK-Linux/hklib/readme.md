## 库文件加载说明
1. lib文件夹里面所有库文件libhcnetsdk.so、libHCCore.so、libssl.so、libcrypto.so、libcrypto.so.1.0.0以及HCNetSDKCom文件夹都需要加载到工程中。

2. HCNetSDKCom文件夹如果和libhcnetsdk.so、libhpr.so、libHCCore文件、可执行文件不在同级目录，或者加载失败，可以调用NET_DVR_SetSDKInitCfg(enumType类型赋值为2，lpInBuff对应结构体NET_DVR_LOCAL_SDK_PATH)设置组件库所在路径。

3. libcrypto.so、libcrypto.so.1.0.0和libssl.so是开源库，如果库文件加载失败，可以调用NET_DVR_SetSDKInitCfg(enumType类型赋值为3，lpInBuff对应libcrypto.so所在的路径字符串)、NET_DVR_SetSDKInitCfg(enumType类型赋值为4，lpInBuff对应libssl.so所在的路径字符串)指定下这些库文件加载路径。

【路径设置的Java示例代码】
//这里是库的绝对路径，请根据实际情况修改，注意改路径必须有访问权限

//设置HCNetSDKCom组件库所在路径		
String strPathCom = "/home/hik/Desktop/JavaDemoLinux64/lib/HCNetSDKCom";
HCNetSDK.NET_DVR_LOCAL_SDK_PATH struComPath = new HCNetSDK.NET_DVR_LOCAL_SDK_PATH();
System.arraycopy(strPathCom.getBytes(), 0, struComPath.sPath, 0, strPathCom.length());
struComPath.write();
hCNetSDK.NET_DVR_SetSDKInitCfg(2, struComPath.getPointer());

//设置libcrypto.so所在路径	
HCNetSDK.BYTE_ARRAY ptrByteArrayCrypto = new HCNetSDK.BYTE_ARRAY(256);
String strPathCrypto = "/home/hik/Desktop/JavaDemoLinux64/lib/libcrypto.so";		
System.arraycopy(strPathCrypto.getBytes(), 0, ptrByteArrayCrypto.byValue, 0, strPathCrypto.length());
ptrByteArrayCrypto.write();
hCNetSDK.NET_DVR_SetSDKInitCfg(3, ptrByteArrayCrypto.getPointer());

//设置libssl.so所在路径	
HCNetSDK.BYTE_ARRAY ptrByteArraySsl = new HCNetSDK.BYTE_ARRAY(256);	
String strPathSsl = "/home/hik/Desktop/JavaDemoLinux64/lib/libssl.so";	
System.arraycopy(strPathSsl.getBytes(), 0, ptrByteArraySsl.byValue, 0, strPathSsl.length());
ptrByteArraySsl.write();
hCNetSDK.NET_DVR_SetSDKInitCfg(4, ptrByteArraySsl.getPointer());


## Version
`CH-HCNetSDKV6.0.2.35_build20190411_Linux64`