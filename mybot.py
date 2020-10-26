from engine.pyalice import Application

app = Application(name = "mybot")
app.load_module('message_generators')
app.load_module('viewers')

app.load('packages/navsim/apps/navsim_tcp.subgraph.json', 'simulation')

node_view = app.add("viewer")
component_view = node_view.add(app.registry.isaac.viewers.ColorCameraViewer, 'ColorCameraViewer')

node_view = app.add("depth_viewer")
node_view.add(app.registry.isaac.viewers.DepthCameraViewer, 'DepthCameraViewer')

app.connect('simulation.interface/output', 'color', 'viewer/ColorCameraViewer', 'color_listener')
app.connect('simulation.interface/output', 'depth', 'depth_viewer/DepthCameraViewer', 'depth_listener')
app.run()