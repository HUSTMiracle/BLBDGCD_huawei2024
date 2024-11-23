import os
import ctypes
import functools
from .hk_define import *
from .hk_struct import LPNET_DVR_DEVICEINFO_V30, NET_DVR_FOCUSMODE_CFG, NET_DVR_JPEGPARA
from .hikvision_infrared import get_temper_info


# 禁止使用logging模块


def _release_wrapper(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        res = func(*args, **kwargs)
        if kwargs.get('release_resources', True):
            if args[0].user_id != -1:
                args[0]._destroy()
        return res

    return inner


class HIKVisionSDK(object):
    def __init__(self, lib_dir, ip, username, password, port=8000, channel=1, debug=True):
        self.lib_dir = lib_dir
        self.old_cwd = os.getcwd()
        self.ip = ip
        self.username = username
        self.password = password
        self.port = port
        self.user_id = -1
        self.hk_so_lib = None
        self.channel = channel
        self.err_code = 0
        self.debug = debug

    def print_log(self, msg):
        if self.debug:
            print(msg)

    def init(self):
        """raise a exception if error"""
        self.print_log('开始改变工作目录 %s' % self.lib_dir)
        os.chdir(self.lib_dir)
        self.print_log('开始加载libhcnetsdk.so')
        self.hk_so_lib = ctypes.cdll.LoadLibrary("./libhcnetsdk.so")
        ok = self.hk_so_lib.NET_DVR_Init()
        if not ok:
            self.err_code = -1
            raise Exception("<<<海康sdk初始化失败")
        self._login()
        return self

    def _login(self):
        self.print_log('开始登录')
        device_info = LPNET_DVR_DEVICEINFO_V30()
        result = self.hk_so_lib.NET_DVR_Login_V30(bytes(self.ip, 'ascii'),
                                                  self.port,
                                                  bytes(self.username, 'ascii'),
                                                  bytes(self.password, 'ascii'),
                                                  ctypes.byref(device_info))
        if result == -1:
            error_num = self.hk_so_lib.NET_DVR_GetLastError()
            self.err_code = error_num
            self._destroy(logout=False)
            raise Exception("<<<海康SDK调用错误 ERRCODE: %s" % error_num)
        self.print_log('登录成功')
        self.user_id = result

    def _destroy(self, logout=True):
        if logout:
            self.print_log('>>>开始注销资源')
            res = self.hk_so_lib.NET_DVR_Logout(self.user_id)
            if not res:
                self.print_log('<<<User退出失败')
        self.print_log('>>>开始释放资源')
        res = self.hk_so_lib.NET_DVR_Cleanup()
        if not res:
            self.print_log('<<<释放资源失败')
        os.chdir(self.old_cwd)
        self.print_log('>>>成功还原工作目录 %s' % os.getcwd())

    @_release_wrapper
    def take_picture(self, pic_pathname, release_resources=True) -> bool:
        if self.user_id == -1:
            self.print_log('未初始化或者初始化失败')
            return False
        self.print_log('开始拍照 %s' % pic_pathname)
        obj = NET_DVR_JPEGPARA()
        result = self.hk_so_lib.NET_DVR_CaptureJPEGPicture(self.user_id,
                                                           self.channel,
                                                           ctypes.byref(obj),
                                                           bytes(pic_pathname, 'utf-8'))

        if not result:
            error_num = self.hk_so_lib.NET_DVR_GetLastError()
            self.print_log('<<<拍照失败 ERRCODE: %s' % error_num)
            return False
        self.print_log('拍照成功')
        return True

    @_release_wrapper
    def get_zoom(self, release_resources=True) -> int:
        """-1 if failure"""
        if self.user_id == -1:
            self.print_log('<<<未初始化或者初始化失败 user_id %s' % self.user_id)
            return False
        self.print_log('开始获取变焦')
        struct_cfg = NET_DVR_FOCUSMODE_CFG()
        dw_returned = ctypes.c_uint16(0)
        result = self.hk_so_lib.NET_DVR_GetDVRConfig(self.user_id,
                                                     NET_DVR_GET_FOCUSMODECFG,
                                                     self.channel,
                                                     ctypes.byref(struct_cfg),
                                                     255,
                                                     ctypes.byref(dw_returned))
        if not result:
            self.print_log('<<<获取变焦失败')
            return -1
        self.print_log('value %s' % struct_cfg.fOpticalZoomLevel)
        return struct_cfg.fOpticalZoomLevel

    @_release_wrapper
    def set_zoom(self, zoom, release_resources=True) -> bool:
        if self.hk_so_lib == -1:
            self.print_log('<<<未初始化或者初始化失败')
            return False
        self.print_log('开始设置变倍 zoom %s' % zoom)
        struct_cfg = NET_DVR_FOCUSMODE_CFG()
        dw_returned = ctypes.c_uint16(0)
        result = self.hk_so_lib.NET_DVR_GetDVRConfig(self.user_id,
                                                     NET_DVR_GET_FOCUSMODECFG,
                                                     self.channel,
                                                     ctypes.byref(struct_cfg),
                                                     255,
                                                     ctypes.byref(dw_returned))
        if not result:
            self.print_log('<<<获取变倍失败')
            return False
        cur_zoom = struct_cfg.fOpticalZoomLevel
        self.print_log("当前变倍值为 {} ".format(cur_zoom))

        if cur_zoom == zoom:
            self.print_log('已经是相同的倍值 %s' % cur_zoom)
            return True

        if cur_zoom == 0:
            self.print_log('此摄像头不支持变焦')
            return False

        struct_cfg.fOpticalZoomLevel = ctypes.c_float(zoom)
        result = self.hk_so_lib.NET_DVR_SetDVRConfig(self.user_id,
                                                     NET_DVR_SET_FOCUSMODECFG,
                                                     self.channel,
                                                     ctypes.byref(struct_cfg),
                                                     255)
        if not result:
            self.print_log('<<<变倍失败')
            return False
        self.print_log('success %s' % zoom)
        return True

    def get_infrared_value(self) -> tuple:
        os.chdir(self.lib_dir)
        self.print_log('开始获取红外')
        try:
            min_temper, max_temper, aver_temp = get_temper_info(ip=self.ip, username=self.username,
                                                                password=self.password)
        except Exception as e:
            self.print_log(e)
            min_temper, max_temper, aver_temp = -1, -1, -1
        self.print_log(" min_temper {0}, max_temper {1}, aver_temp {2}".format(min_temper, max_temper, aver_temp))
        os.chdir(self.old_cwd)
        return min_temper, max_temper, aver_temp
