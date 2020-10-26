load("//engine/build:isaac.bzl", "isaac_py_app")

isaac_py_app(
    name = "sim_cv", 
    srcs = ["sim_cv.py"],
    data = [
        "//packages/navsim/apps:navsim_tcp_subgraph",
    ],
    modules = [
        "message_generators",
        "viewers",
    ],
    deps = [
        "//engine/pyalice",
    ],
)