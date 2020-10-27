from engine.pyalice import *
import numpy as np
import cv2 as cv

class SimCv(Codelet):
    def start(self):
        self.rx_color = self.isaac_proto_rx("ColorCameraProto", 'input_image')
        self.rx_depth = self.isaac_proto_rx("DepthCameraProto", 'depth_image')
        self.tick_on_message(self.rx_color)
        # self.tick_periodically(0.1)


    def tick(self):
        rx_message = self.rx_color.message
        rx_depth_message = self.rx_depth.message

        # rx_message = app.receive('', 'MessageLedger', 'out')
        if rx_message is None or rx_depth_message is None:
            return

        #获取ColorCameraProto数据的信息
        img_buf = np.asarray(rx_message.buffers[0])
        rows = rx_message.proto.image.rows
        cols = rx_message.proto.image.cols
        channels = rx_message.proto.image.channels
        buffer_index = rx_message.proto.image.dataBufferIndex
        print(f"rows:{rows}, cols:{cols}, channels:{channels}, index:{buffer_index}")
        print(img_buf.shape)
        #显示color image
        img = img_buf.reshape(rows, cols, channels)[:, :, ::-1]
        img = cv.resize(img, (640, 360))
        cv.imshow('curl', img)
        cv.waitKey(1)

        print("-"*10)

        #获取DepthCameraProto数据信息
        rows_d = rx_depth_message.proto.depthImage.rows
        cols_d = rx_depth_message.proto.depthImage.cols
        channels_d = rx_depth_message.proto.depthImage.channels
        buffer_index_d = rx_depth_message.proto.depthImage.dataBufferIndex
        print(f"rows:{rows_d}, cols:{cols_d}, channels:{channels_d}, index:{buffer_index_d}")
        print(rx_depth_message.tensor.shape) # tensor是numpy中的ndarray
        #显示depth image
        img_depth_buf = rx_depth_message.tensor/20 #最大值20.0
        img_depth_buf = cv.resize(img_depth_buf, (640, 360))
        cv.imshow('curl_depth', img_depth_buf)
        cv.waitKey(1)

        print("-"*80)
     
# class SimSegmentation(Codelet):
#     def start(self):
#         self.seg_rx = self.isaac_proto_rx("SegmentationCameraProto", 'segmentation_image')

class SimLidar(Codelet):
    def start(self):
        self.lidar_rx = self.isaac_proto_rx('RangeScanProto', 'range_scan')
        self.tick_on_message(self.lidar_rx)

    def tick(self):
        lidar_message = self.lidar_rx.message
        if lidar_message is None:
            return
        lidar_tensor = lidar_message.tensor
        ranges = lidar_message.proto.ranges
        intensities = lidar_message.proto.intensities

        range_sizes = ranges.sizes
        intensities_sizes = intensities.sizes
        print(lidar_tensor.shape)
        print(range_sizes, intensities_sizes)
        print(ranges.dataBufferIndex, intensities.dataBufferIndex)

        # print(dir(lidar_message))
        # print(dir(ranges))
        # print(dir(intensities))
        print('-'*80)


def main():
    app = Application(name = 'sim_cv')
    app.load_module('viewers')
    app.load('packages/navsim/apps/navsim_tcp.subgraph.json', 'simulation')
    node_view = app.add('viewer')
    component_color_view = node_view.add(app.registry.isaac.viewers.ColorCameraViewer, 'ColorCameraViewer')
    component_depth_view = node_view.add(app.registry.isaac.viewers.DepthCameraViewer, 'DepthCameraViewer')
    component_seg_view = node_view.add(app.registry.isaac.viewers.SegmentationCameraViewer, 'SegmentationCameraViewer')
    component_seg_view.config.camera_name = 'SegmentationCamera'

    #添加test 节点，添加component
    node_test = app.add('test')
    component_color_test = node_test.add(SimCv, 'SimCvColor')
    component_depth_test = node_test.add(SimCv, 'SimCvDepth')

    node_lidar = app.add('lidar')
    component_lidar = node_lidar.add(SimLidar, 'SimLidar')

    app.connect('simulation.interface/output', 'color', 'viewer/ColorCameraViewer', 'color_listener')
    app.connect('simulation.interface/output', 'depth', 'viewer/DepthCameraViewer', 'depth_listener')
    app.connect('simulation.interface/output', 'segmentation', 'viewer/SegmentationCameraViewer', 'segmentation_listener')
    # app.connect('simulation.interface/output', 'color', 'test/SimCvColor', 'input_image')
    # app.connect('simulation.interface/output', 'depth', 'test/SimCvColor', 'depth_image')

    app.connect('simulation.interface/output', 'rangescan', 'lidar/SimLidar', 'range_scan')
    app.run()

if __name__=='__main__':
    main()
