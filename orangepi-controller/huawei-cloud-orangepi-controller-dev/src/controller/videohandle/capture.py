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


def load_so(so_dir):
    
    # os.chdir(so_dir)
    lib_hc_net_sdk = ctypes.cdll.LoadLibrary(so_dir + "/libhcnetsdk.so")
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

    # 拍照
    name = "/home/orangepi/huawei-cloud-orangepi-controller/src/HCNetPythonSDK-Linux/test/test.jpg"
    obj = NET_DVR_JPEGPARA()
    obj.wPicQuality = 0
    obj.wPicSize = 16
    res = lib_hc_net_sdk.NET_DVR_CaptureJPEGPicture(l_user_id, l_channel, ctypes.byref(obj), bytes(name, 'utf-8'))
    if not res:
        print('拍照失败')
    else:
        print('拍照成功 %s ' % name)

    # 释放资源
    res = lib_hc_net_sdk.NET_DVR_Logout(l_user_id)
    if not res:
        print('退出失败', lib_hc_net_sdk.NET_DVR_GetLastError())
    res = lib_hc_net_sdk.NET_DVR_Cleanup()
    print('NET_DVR_Cleanup', res)
    print('NET_DVR_Logout', res)


load_so('/home/orangepi/huawei-cloud-orangepi-controller/src/HCNetPythonSDK-Linux/hklib')
