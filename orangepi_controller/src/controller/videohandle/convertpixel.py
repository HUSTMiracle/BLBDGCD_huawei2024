import os
import sys
import numpy as np
from os import getcwd
import cv2
import time

from ctypes import *



sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *


def enum_devices(device=0, device_way=False):
    if device_way == False:
        if device == 0:
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_UNKNOW_DEVICE | MV_1394_DEVICE | MV_CAMERALINK_DEVICE
            deviceList = MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
            if ret != 0:
                print("enum devices fail! ret[0x%x]" % ret)
                sys.exit()
            if deviceList.nDeviceNum == 0:
                print("find no device!")
                sys.exit()
            print("Find %d devices!" % deviceList.nDeviceNum)
            return deviceList
        else:
            pass
    elif device_way == True:
        pass

def identify_different_devices(deviceList):
    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            nip1_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip1_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip1_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip1_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            nip2_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0xff000000) >> 24)
            nip2_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0x00ff0000) >> 16)
            nip2_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0x0000ff00) >> 8)
            nip2_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0x000000ff)
            nip3_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0xff000000) >> 24)
            nip3_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0x00ff0000) >> 16)
            nip3_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0x0000ff00) >> 8)
            nip3_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0x000000ff)
            nip4_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0xff000000) >> 24)
            nip4_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0x00ff0000) >> 16)
            nip4_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0x0000ff00) >> 8)
            nip4_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0x000000ff)
            strmanufacturerName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chManufacturerName:
                strmanufacturerName = strmanufacturerName + chr(per)
            stdeviceversion = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chDeviceVersion:
                stdeviceversion = stdeviceversion + chr(per)
            stManufacturerSpecificInfo = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chManufacturerSpecificInfo:
                stManufacturerSpecificInfo = stManufacturerSpecificInfo + chr(per)
            stSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber:
                stSerialNumber = stSerialNumber + chr(per)
            stUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                stUserDefinedName = stUserDefinedName + chr(per)
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            strmanufacturerName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chVendorName:
                strmanufacturerName = strmanufacturerName + chr(per)
            stdeviceversion = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chDeviceVersion:
                stdeviceversion = stdeviceversion + chr(per)
            stSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                stSerialNumber = stSerialNumber + chr(per)
            stUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                stUserDefinedName = stUserDefinedName + chr(per)
            stDeviceGUID = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chDeviceGUID:
                stDeviceGUID = stDeviceGUID + chr(per)
            stFamilyName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chFamilyName:
                stFamilyName = stFamilyName + chr(per)
        elif mvcc_dev_info.nTLayerType == MV_1394_DEVICE:
            print("\n1394-a/b device: [%d]" % i)

        elif mvcc_dev_info.nTLayerType == MV_CAMERALINK_DEVICE:
            print("\ncameralink device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            strmanufacturerName = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chVendorName:
                strmanufacturerName = strmanufacturerName + chr(per)
            stdeviceversion = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chDeviceVersion:
                stdeviceversion = stdeviceversion + chr(per)
def input_num_camera(deviceList):
    return 0
def creat_camera(deviceList, nConnectionNum, log=True, log_path=getcwd()):
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
    if log == True:
        ret = cam.MV_CC_SetSDKLogPath(log_path)
        print(log_path)
        if ret != 0:
            print("set Log path  fail! ret[0x%x]" % ret)
            sys.exit()

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()
    elif log == False:
        ret = cam.MV_CC_CreateHandleWithoutLog(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()
    return cam, stDeviceList


def open_device(cam):
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

def get_Value(cam, param_type="int_value", node_name="PayloadSize"):
    if param_type == "int_value":
        stParam = MVCC_INTVALUE_EX()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
        ret = cam.MV_CC_GetIntValueEx(node_name, stParam)
        if ret != 0:
            sys.exit()
        int_value = stParam.nCurValue
        return int_value

    elif param_type == "float_value":
        stFloatValue = MVCC_FLOATVALUE()
        memset(byref(stFloatValue), 0, sizeof(MVCC_FLOATVALUE))
        ret = cam.MV_CC_GetFloatValue(node_name, stFloatValue)
        if ret != 0:
            sys.exit()
        float_value = stFloatValue.fCurValue
        return float_value

    elif param_type == "enum_value":
        stEnumValue = MVCC_ENUMVALUE()
        memset(byref(stEnumValue), 0, sizeof(MVCC_ENUMVALUE))
        ret = cam.MV_CC_GetEnumValue(node_name, stEnumValue)
        if ret != 0:
            sys.exit()
        enum_value = stEnumValue.nCurValue
        return enum_value

    elif param_type == "bool_value":
        stBool = c_bool(False)
        ret = cam.MV_CC_GetBoolValue(node_name, stBool)
        if ret != 0:
            sys.exit()
        return stBool.value

    elif param_type == "string_value":
        stStringValue = MVCC_STRINGVALUE()
        memset(byref(stStringValue), 0, sizeof(MVCC_STRINGVALUE))
        ret = cam.MV_CC_GetStringValue(node_name, stStringValue)
        if ret != 0:
            sys.exit()
        string_value = stStringValue.chCurValue
        return string_value

def set_Value(cam, param_type="int_value", node_name="PayloadSize", node_value=None):
    if param_type == "int_value":
        stParam = int(node_value)
        ret = cam.MV_CC_SetIntValueEx(node_name, stParam)
        if ret != 0:
            sys.exit()

    elif param_type == "float_value":
        stFloatValue = float(node_value)
        ret = cam.MV_CC_SetFloatValue(node_name, stFloatValue)
        if ret != 0:
            sys.exit()

    elif param_type == "enum_value":
        
        
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        if ret != 0:
            sys.exit()
        

    elif param_type == "bool_value":
        ret = cam.MV_CC_SetBoolValue(node_name, node_value)
        if ret != 0:
            
            sys.exit()
        

    elif param_type == "string_value":
        stStringValue = str(node_value)
        ret = cam.MV_CC_SetStringValue(node_name, stStringValue)
        if ret != 0:
            
            sys.exit()

def read_or_write_memory(cam, way="read"):
    if way == "read":
        pass
        cam.MV_CC_ReadMemory()
    elif way == "write":
        pass
        cam.MV_CC_WriteMemory()

def decide_divice_on_line(cam):
    value = cam.MV_CC_IsDeviceConnected()



def set_image_Node_num(cam, Num=1):
    ret = cam.MV_CC_SetImageNodeNum(nNum=Num)


def set_grab_strategy(cam, grabstrategy=0, outputqueuesize=1):
    if grabstrategy != 2:
        ret = cam.MV_CC_SetGrabStrategy(enGrabStrategy=grabstrategy)
        
    else:
        ret = cam.MV_CC_SetGrabStrategy(enGrabStrategy=grabstrategy)
        ret = cam.MV_CC_SetOutputQueueSize(nOutputQueueSize=outputqueuesize)
        
        
        
        



def image_show(image, name):
    image = cv2.resize(image, (1000, 700), interpolation=cv2.INTER_AREA)
    name = str(name)

    
    
    

    
    
    cv2.imshow(name, image)
    k = cv2.waitKey(1) & 0xff



def image_control(data, stFrameInfo,path):
    if stFrameInfo.enPixelType == 17301513:   
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BayerRG2RGB)
        cv2.imwrite(path, image)




def access_get_image(cam, path,active_way="getImagebuffer"):

    if active_way == "getImagebuffer":
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret and stOutFrame.stFrameInfo.enPixelType == 17301513:
          print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
          stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
          pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
          libc = CDLL('libc.so.6')   
          libc.memcpy(byref(pData), stOutFrame.pBufAddr,stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)   
                
          data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),
                                     dtype=np.uint8)
          image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo,path=path)


    elif active_way == "getoneframetimeout":
        stParam = MVCC_INTVALUE_EX()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
        ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()
        nDataSize = stParam.nCurValue
        pData = (c_ubyte * nDataSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d] " % (
                stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
                image = np.asarray(pData)
                image_control(data=image, stFrameInfo=stFrameInfo)
            else:
                print("no data[0x%x]" % ret)



winfun_ctype = CFUNCTYPE
stFrameInfo = POINTER(MV_FRAME_OUT_INFO_EX)
pData = POINTER(c_ubyte)
FrameInfoCallBack = winfun_ctype(None, pData, stFrameInfo, c_void_p)


def image_callback(pData, pFrameInfo, pUser):
    global img_buff
    img_buff = None
    stFrameInfo = cast(pFrameInfo, POINTER(MV_FRAME_OUT_INFO_EX)).contents
    if stFrameInfo:
        print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
        stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
    if img_buff is None and stFrameInfo.enPixelType == 17301505:
        img_buff = (c_ubyte * stFrameInfo.nWidth * stFrameInfo.nHeight)()
        cdll.msvcrt.memcpy(byref(img_buff), pData, stFrameInfo.nWidth * stFrameInfo.nHeight)
        data = np.frombuffer(img_buff, count=int(stFrameInfo.nWidth * stFrameInfo.nHeight), dtype=np.uint8)
        image_control(data=data, stFrameInfo=stFrameInfo)
        del img_buff
    elif img_buff is None and stFrameInfo.enPixelType == 17301514:
        img_buff = (c_ubyte * stFrameInfo.nWidth * stFrameInfo.nHeight)()
        cdll.msvcrt.memcpy(byref(img_buff), pData, stFrameInfo.nWidth * stFrameInfo.nHeight)
        data = np.frombuffer(img_buff, count=int(stFrameInfo.nWidth * stFrameInfo.nHeight), dtype=np.uint8)
        image_control(data=data, stFrameInfo=stFrameInfo)
        del img_buff
    elif img_buff is None and stFrameInfo.enPixelType == 35127316:
        img_buff = (c_ubyte * stFrameInfo.nWidth * stFrameInfo.nHeight * 3)()
        cdll.msvcrt.memcpy(byref(img_buff), pData, stFrameInfo.nWidth * stFrameInfo.nHeight * 3)
        data = np.frombuffer(img_buff, count=int(stFrameInfo.nWidth * stFrameInfo.nHeight * 3), dtype=np.uint8)
        image_control(data=data, stFrameInfo=stFrameInfo)
        del img_buff
    elif img_buff is None and stFrameInfo.enPixelType == 34603039:
        img_buff = (c_ubyte * stFrameInfo.nWidth * stFrameInfo.nHeight * 2)()
        cdll.msvcrt.memcpy(byref(img_buff), pData, stFrameInfo.nWidth * stFrameInfo.nHeight * 2)
        data = np.frombuffer(img_buff, count=int(stFrameInfo.nWidth * stFrameInfo.nHeight * 2), dtype=np.uint8)
        image_control(data=data, stFrameInfo=stFrameInfo)
        del img_buff


CALL_BACK_FUN = FrameInfoCallBack(image_callback)

stEventInfo = POINTER(MV_EVENT_OUT_INFO)
pData = POINTER(c_ubyte)
EventInfoCallBack = winfun_ctype(None, stEventInfo, c_void_p)


def event_callback(pEventInfo, pUser):
    stPEventInfo = cast(pEventInfo, POINTER(MV_EVENT_OUT_INFO)).contents
    nBlockId = stPEventInfo.nBlockIdHigh
    nBlockId = (nBlockId << 32) + stPEventInfo.nBlockIdLow
    nTimestamp = stPEventInfo.nTimestampHigh
    nTimestamp = (nTimestamp << 32) + stPEventInfo.nTimestampLow
    if stPEventInfo:
        print("EventName[%s], EventId[%u], BlockId[%d], Timestamp[%d]" % (
        stPEventInfo.EventName, stPEventInfo.nEventID, nBlockId, nTimestamp))


CALL_BACK_FUN_2 = EventInfoCallBack(event_callback)


def call_back_get_image(cam):
    ret = cam.MV_CC_RegisterImageCallBackEx(CALL_BACK_FUN, None)
    if ret != 0:
        print("register image callback fail! ret[0x%x]" % ret)
        sys.exit()

def close_and_destroy_device(cam, data_buf=None):
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
    del data_buf


def start_grab_and_get_data_size(cam):
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:

        sys.exit()


def picture(path):

    deviceList = enum_devices(device=0, device_way=False)

    identify_different_devices(deviceList)

    nConnectionNum = input_num_camera(deviceList)

    cam, stDeviceList = creat_camera(deviceList, nConnectionNum, log=False)
    
    open_device(cam)
    
    # 加载相机参数
    ret = cam.MV_CC_FeatureLoad("/home/orangepi/huawei-cloud-orangepi-controller/src/controller/videohandle/FeatureFile.ini")

    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    start_grab_and_get_data_size(cam)
    access_get_image(cam, active_way="getImagebuffer",path=path)
    close_and_destroy_device(cam)


if __name__ == "__main__":
    while 1:
        path = "test_onnx.bmp"
        picture(path)
        time.sleep(1)
