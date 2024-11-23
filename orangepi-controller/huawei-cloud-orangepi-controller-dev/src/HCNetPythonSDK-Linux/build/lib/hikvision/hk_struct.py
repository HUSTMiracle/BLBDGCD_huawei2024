import ctypes


# 登录需要的结构体
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


# 拍照结构体
class NET_DVR_JPEGPARA(ctypes.Structure):
    _fields_ = [
        ("wPicSize", ctypes.c_ushort),
        ("wPicQuality", ctypes.c_ushort)]