"""
1 初始化SDK资源
bool NET_DVR_Init()
2 用户登录
LONG NET_DVR_Login_V30(LPNET_DVR_USER_LOGIN_INFO pLoginInfo, LPNET_DVR_DEVICEINFO_V30 lpDeviceInfo)
3 获取参数
 BOOL NET_DVR_GetDVRConfig(LONG lUserID, DWORD dwCommand,LONG lChannel, LPVOID lpOutBuffer, DWORD dwOutBufferSize,
 LPDWORD lpBytesReturned)

4 设置参数
BOOL NET_DVR_SetDVRConfig(LONG lUserID, DWORD dwCommand,LONG lChannel, LPVOID
lpInBuffer, DWORD dwInBufferSize)

#define NET_DVR_GET_FOCUSMODECFG                3305//获取快球聚焦模式信息

5 注销用户
6 释放SDK资源
"""

import ctypes
import os
import random

l_user_id = None
l_channel = 1
user_name = 'admin'
user_password = 'IA5cWJEY'
camera_ip = '202.114.213.56'
port = 8000
NET_DVR_GET_FOCUSMODECFG = 3305
NET_DVR_SET_FOCUSMODECFG = 3306

print(os.getcwd())


class LPNET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_byte * 48),
        ("byAlarmInPortNum", ctypes.c_byte),
        ("byAlarmOutPortNum", ctypes.c_byte),
        ("byDiskNum", ctypes.c_byte),
        ("byDVRType", ctypes.c_byte),
        ("byChanNum", ctypes.c_byte),
        ("byStartChan", ctypes.c_byte),
        ("byAudioChanNum", ctypes.c_byte),
        ("byIPChanNum", ctypes.c_byte),
        ("byZeroChanNum", ctypes.c_byte),
        ("byMainProto", ctypes.c_byte),
        ("bySubProto", ctypes.c_byte),
        ("bySupport", ctypes.c_byte),
        ("bySupport1", ctypes.c_byte),
        ("bySupport2", ctypes.c_byte),
        ("wDevType", ctypes.c_uint16),
        ("bySupport3", ctypes.c_byte),
        ("byMultiStreamProto", ctypes.c_byte),
        ("byStartDChan", ctypes.c_byte),
        ("byStartDTalkChan", ctypes.c_byte),
        ("byHighDChanNum", ctypes.c_byte),
        ("bySupport4", ctypes.c_byte),
        ("byLanguageType", ctypes.c_byte),
        ("byVoiceInChanNum", ctypes.c_byte),
        ("byStartVoiceInChanNo", ctypes.c_byte),
        ("byRes3", ctypes.c_byte * 2),
        ("byMirrorChanNum", ctypes.c_byte),
        ("wStartMirrorChanNo", ctypes.c_uint16),
        ("byRes2", ctypes.c_byte * 2)]


# 光学变倍结构体
class NET_DVR_FOCUSMODE_CFG(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint16),
        ("byFocusMode", ctypes.c_byte),
        ("byAutoFocusMode", ctypes.c_byte),
        ("wMinFocusDistance", ctypes.c_uint16),
        ("byZoomSpeedLevel", ctypes.c_byte),
        ("byFocusSpeedLevel", ctypes.c_byte),
        ("byOpticalZoom", ctypes.c_byte),
        ("byDigtitalZoom", ctypes.c_byte),
        ("fOpticalZoomLevel", ctypes.c_float),
        ("dwFocusPos", ctypes.c_uint16),
        ("byFocusDefinitionDisplay", ctypes.c_byte),
        ("byFocusSensitivity", ctypes.c_byte),
        ("byRes1", ctypes.c_byte * 2),
        ("dwRelativeFocusPos", ctypes.c_uint16),
        ("byRes", ctypes.c_byte * 48), ]

def test_load():
    pass


class NET_DVR_JPEGPARA(ctypes.Structure):
    _fields_ = [
        ("wPicSize", ctypes.c_ushort),
        ("wPicQuality", ctypes.c_ushort)]

class LPNET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_byte * 48),
        ("byAlarmInPortNum", ctypes.c_byte),
        ("byAlarmOutPortNum", ctypes.c_byte),
        ("byDiskNum", ctypes.c_byte),
        ("byDVRType", ctypes.c_byte),
        ("byChanNum", ctypes.c_byte),
        ("byStartChan", ctypes.c_byte),
        ("byAudioChanNum", ctypes.c_byte),
        ("byIPChanNum", ctypes.c_byte),
        ("byZeroChanNum", ctypes.c_byte),
        ("byMainProto", ctypes.c_byte),
        ("bySubProto", ctypes.c_byte),
        ("bySupport", ctypes.c_byte),
        ("bySupport1", ctypes.c_byte),
        ("bySupport2", ctypes.c_byte),
        ("wDevType", ctypes.c_uint16),
        ("bySupport3", ctypes.c_byte),
        ("byMultiStreamProto", ctypes.c_byte),
        ("byStartDChan", ctypes.c_byte),
        ("byStartDTalkChan", ctypes.c_byte),
        ("byHighDChanNum", ctypes.c_byte),
        ("bySupport4", ctypes.c_byte),
        ("byLanguageType", ctypes.c_byte),
        ("byVoiceInChanNum", ctypes.c_byte),
        ("byStartVoiceInChanNo", ctypes.c_byte),
        ("byRes3", ctypes.c_byte * 2),
        ("byMirrorChanNum", ctypes.c_byte),
        ("wStartMirrorChanNo", ctypes.c_uint16),
        ("byRes2", ctypes.c_byte * 2)]

def load_sdk():
    sos = os.listdir('/home/orangepi/huawei-cloud-orangepi-controller/src/HCNetPythonSDK-Linux/hklib/HCNetSDKCom')
    so_path = []
    for i in sos:
        so_path.append('/home/orangepi/huawei-cloud-orangepi-controller/src/HCNetPythonSDK-Linux/hklib/HCNetSDKCom/'+i)

    # for i in so_path:
    #     print(ctypes.cdll.LoadLibrary(i))

    sos = os.listdir('/home/orangepi/huawei-cloud-orangepi-controller/src/HCNetPythonSDK-Linux/hklib/lib')
    so2_path = []
    for i in sos:
        if str(i).endswith('.so'):
            so2_path.append('/home/orangepi/huawei-cloud-orangepi-controller/src/HCNetPythonSDK-Linux/hklib/lib/'+i)

    print('soss', so2_path)
    for i in so2_path:
        print(ctypes.cdll.LoadLibrary(i))
    # ctypes.cdll.LoadLibrary('/home/baloneo/PythonProject/HCNetPythonSDK-Linux/lib/libAudioRender.so')


def load_so(so_dir):
    print('切换工作目录和库文件所在路径：', so_dir)
    os.chdir(so_dir)
    # lib_hc_core = ctypes.cdll.LoadLibrary("./libHCCore.so")
    # print(dir(lib_hc_core))
    # lib_hc_core.NET_DVR_Init()
    # ctypes.cdll.LoadLibrary("/home/baloneo/PythonProject/HCNetPythonSDK-Linux/lib/libcrypto.so.1.0.0")
    # ctypes.cdll.LoadLibrary("/home/baloneo/PythonProject/HCNetPythonSDK-Linux/lib/libcrypto.so")

    # ctypes.cdll.LoadLibrary("/home/baloneo/PythonProject/HCNetPythonSDK-Linux/lib/libssl.so")

    lib_hc_net_sdk = ctypes.cdll.LoadLibrary("./libhcnetsdk.so")
    ok = lib_hc_net_sdk.NET_DVR_Init()
    print("NET_DVR_Init: ", ok)
    out_device_info = LPNET_DVR_DEVICEINFO_V30()
    l_user_id = lib_hc_net_sdk.NET_DVR_Login_V30(bytes(camera_ip, 'ascii'),
                                                 port,
                                                 bytes(user_name, 'ascii'),
                                                 bytes(user_password, 'ascii'),
                                                 ctypes.byref(out_device_info))
    print('out_device_info', out_device_info)
    if l_user_id == -1:  # 登录失败
        error_num = lib_hc_net_sdk.NET_DVR_GetLastError()
        if error_num == 7:
            print("连接设备失败设备不在线或网络原因引起的连接超时等")
        print('err_num', error_num)
        res = lib_hc_net_sdk.NET_DVR_Cleanup()
        print('NET_DVR_Cleanup', res)
        return
    else:
        print('登录成功')
    print("NET_DVR_Login_V30:", l_user_id)

    # 获取变倍数值
    struct_docus_cfg = NET_DVR_FOCUSMODE_CFG()
    dw_returned = ctypes.c_uint16(0)
    res = lib_hc_net_sdk.NET_DVR_GetDVRConfig(l_user_id,
                                        NET_DVR_GET_FOCUSMODECFG,
                                        l_channel,
                                        ctypes.byref(struct_docus_cfg),
                                        255,
                                        ctypes.byref(dw_returned))
    if not res:
        print('获取变倍失败')
    print('获取到的变焦值:', struct_docus_cfg.fOpticalZoomLevel, "dw_returned", dw_returned)

    # 设置变倍
    new_zoom = 7
    if struct_docus_cfg.fOpticalZoomLevel == new_zoom:
        print('忽略 已经是这个值')
    struct_docus_cfg.fOpticalZoomLevel = new_zoom
    res = lib_hc_net_sdk.NET_DVR_SetDVRConfig(l_user_id,
                                        NET_DVR_SET_FOCUSMODECFG,
                                        l_channel,
                                        ctypes.byref(struct_docus_cfg),
                                        255)
    name = "/home/orangepi/huawei-cloud-orangepi-controller/src/HCNetPythonSDK-Linux/test/test.jpg"
    obj = NET_DVR_JPEGPARA()
    obj.wPicQuality = 0
    obj.wPicSize = 26
    res = lib_hc_net_sdk.NET_DVR_CaptureJPEGPicture(l_user_id, l_channel, ctypes.byref(obj), bytes(name, 'utf-8'))
    if not res:
        print('拍照失败')
    else:
        print('拍照成功 %s ' % name)

    if not res:
        print('变倍失败')
    print('变倍成功 值为', struct_docus_cfg.fOpticalZoomLevel)

    # 释放资源
    res = lib_hc_net_sdk.NET_DVR_Logout(l_user_id)
    if not res:
        print('退出失败', lib_hc_net_sdk.NET_DVR_GetLastError())
    res = lib_hc_net_sdk.NET_DVR_Cleanup()
    print('NET_DVR_Cleanup', res)
    print('NET_DVR_Logout', res)


# load_sdk()
load_so('/home/orangepi/huawei-cloud-orangepi-controller/src/HCNetPythonSDK-Linux/hklib')
