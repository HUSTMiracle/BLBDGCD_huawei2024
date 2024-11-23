import sys
import os
import cv2
import gi
import numpy as np
import torch

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

class Pipeline:
    def __init__(self, device_path, model_path, output_dir):
        self.device_path = device_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.pipeline = None
        self.loop = None
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='local')  # Load model for CPU

    def detect_and_save(self, img, frame_number):
        results = self.model(img)
        for idx, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
            x1, y1, x2, y2 = map(int, xyxy)
            cropped_img = img[y1:y2, x1:x2]

            output_path = os.path.join(self.output_dir, f"pcb_{frame_number}_{idx}.jpg")
            cv2.imwrite(output_path, cropped_img)
            print(f"Saved cropped image to {output_path}")

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        frame_number = 0
        obj_counter = {
            0: 0,
        }

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer")
            return

        # Extract frame from GstBuffer
        caps = gst_buffer.get_caps()
        size = caps[0].get_size()
        frame_image = np.frombuffer(gst_buffer.extract_dup(0, size), np.uint8).reshape((1080, 1920, 3))  # Adjust dimensions as needed

        # Convert frame to RGB for PyTorch model
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        self.detect_and_save(frame_image, frame_number)

        return Gst.PadProbeReturn.OK

    def start_pipeline(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline()

        source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
        caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
        caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "video_caps")
        sink = Gst.ElementFactory.make("xvimagesink", "video-renderer")  # Use xvimagesink for display

        caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw, format=BGR"))
        source.set_property('device', self.device_path)

        self.pipeline.add(source)
        self.pipeline.add(caps_v4l2src)
        self.pipeline.add(vidconvsrc)
        self.pipeline.add(caps_vidconvsrc)
        self.pipeline.add(sink)

        source.link(caps_v4l2src)
        caps_v4l2src.link(vidconvsrc)
        vidconvsrc.link(caps_vidconvsrc)
        caps_vidconvsrc.link(sink)

        osdsinkpad = sink.get_static_pad("sink")
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline started")

    def stop_pipeline(self):
        self.pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        return True
