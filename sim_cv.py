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

        img_buf = np.asarray(rx_message.buffers[0])
        rows = rx_message.proto.image.rows
        cols = rx_message.proto.image.cols
        channels = rx_message.proto.image.channels
        buffer_index = rx_message.proto.image.dataBufferIndex
        print(f"rows:{rows}, cols:{cols}, channels:{channels}, index:{buffer_index}")
        print(img_buf.shape)
        img = img_buf.reshape(rows, cols, channels)[:, :, ::-1]
        img = cv.resize(img, (640, 360))

        cv.imshow('curl', img)
        cv.waitKey(1)

        print("-"*10)

        rows_d = rx_depth_message.proto.depthImage.rows
        cols_d = rx_depth_message.proto.depthImage.cols
        channels_d = rx_depth_message.proto.depthImage.channels
        buffer_index_d = rx_depth_message.proto.depthImage.channels
        # img_depth = img_depth_buf.reshape(rows_d, cols_d, channels_d)
        print(f"rows:{rows_d}, cols:{cols_d}, channels:{channels_d}, index:{buffer_index_d}")

        img_depth_buf = np.asarray(rx_depth_message.tensor)/255
        img_depth_buf = cv.resize(img_depth_buf, (640, 360))
        cv.imshow('curl_depth', img_depth_buf)
        cv.waitKey(1)
        # print(img_depth_buf.shape)
        # print(img_depth_buf)

        print("-"*80)
     

def main():
    app = Application(name = 'sim_cv')
    app.load_module('viewers')
    app.load('packages/navsim/apps/navsim_tcp.subgraph.json', 'simulation')
    node_view = app.add('viewer')
    component_color_view = node_view.add(app.registry.isaac.viewers.ColorCameraViewer, 'ColorCameraViewer')
    component_depth_view = node_view.add(app.registry.isaac.viewers.DepthCameraViewer, 'DepthCameraViewer')

    #添加print Node，添加Component
    node_test = app.add('test')
    component_color_test = node_test.add(SimCv, 'SimCvColor')
    component_depth_test = node_test.add(SimCv, 'SimCvDepth')

    app.connect('simulation.interface/output', 'color', 'viewer/ColorCameraViewer', 'color_listener')
    app.connect('simulation.interface/output', 'depth', 'viewer/DepthCameraViewer', 'depth_listener')
    app.connect('simulation.interface/output', 'color', 'test/SimCvColor', 'input_image')
    app.connect('simulation.interface/output', 'depth', 'test/SimCvColor', 'depth_image')
    app.run()

if __name__=='__main__':
    main()
