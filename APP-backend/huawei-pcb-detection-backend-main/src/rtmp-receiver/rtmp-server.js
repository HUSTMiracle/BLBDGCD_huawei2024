#!/bin/node
const NodeMediaServer = require('node-media-server');

// Set up the RTMP server
const config = {
    rtmp: {
      port: 1935,
      chunk_size: 60000,
      gop_cache: true,
      ping: 30,
      ping_timeout: 60
    },
    http: {
      port: 8000,
      allow_origin: '*'
    },
    relay: {
        ffmpeg: '/usr/bin/ffmpeg',
        tasks: [
          {
            app: 'cctv',
            mode: 'static',
            edge: 'rtsp://10.12.168.10:554/camera',
            name: '0_149_101',
            rtsp_transport : 'tcp' //['udp', 'tcp', 'udp_multicast', 'http']
          }
        ]
      }
  };
var nms = new NodeMediaServer(config);
nms.run();


