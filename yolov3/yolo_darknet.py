# 导入 numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
# 导入 tvm, relay
import tvm
from tvm import relay
from ctypes import *
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet

import datetime

# 设置 Model name
MODEL_NAME = 'yolov3'

######################################################################
# 下载所需文件
# -----------------------
# 这部分程序下载cfg及weights文件。
# CFG_NAME = MODEL_NAME + '.cfg'
# WEIGHTS_NAME = MODEL_NAME + '.weights'
# REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
# CFG_URL = REPO_URL + 'cfg/' + CFG_NAME + '?raw=true'
# WEIGHTS_URL = 'https://pjreddie.com/media/files/' + WEIGHTS_NAME

# cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")

# 以下为直接填写路径值示例，后同。
cfg_path = "/home/previous/tvm8/yolov3/darknet/yolov3.cfg"

# weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")
weights_path = "/home/previous/tvm8/yolov3/darknet/yolov3.weights"

# 下载并加载DarkNet Library
# if sys.platform in ['linux', 'linux2']:
#     DARKNET_LIB = 'libdarknet2.0.so'
#     DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
# elif sys.platform == 'darwin':
#     DARKNET_LIB = 'libdarknet_mac2.0.so'
#     DARKNET_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'
# else:
#     err = "Darknet lib is not supported on {} platform".format(sys.platform)
#     raise NotImplementedError(err)

# lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")
lib_path = "/home/previous/tvm8/yolov3/darknet/libdarknet2.0.so"

# ******timepoint1-start*******
start1 = datetime.datetime.now()
# ******timepoint1-start*******

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1

data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape_dict = {'data': data.shape}
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

######################################################################
# 将图导入Relay
# -------------------------
# 编译模型
target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {'data': data.shape}
print("Compiling the model...")
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host=target_host,
                                     params=params)


# lib.export_library('deploy_lib.so')

# with open("deploy_graph.json", "w") as fo:
#     fo.write(graph)

# with open("deploy_param.params", "wb") as fo:
#     fo.write(relay.save_param_dict(params))

[neth, netw] = shape['data'][2:] # Current image shape is 608x608

# ******timepoint1-end*******
end1 = datetime.datetime.now()
# ******timepoint1-end*******

######################################################################
# 加载测试图片
# -----------------
test_image = 'dog.jpg'
print("Loading the test image...")
# img_url = REPO_URL + 'data/' + test_image + '?raw=true'
# img_path = download_testdata(img_url, test_image, "data")
img_path = "/home/previous/tvm8/yolov3/darknet/dog.jpg"

# ******timepoint2-start*******
start2 = datetime.datetime.now()
# ******timepoint2-start*******

data = tvm.relay.testing.darknet.load_image(img_path, netw, neth)
######################################################################
# 在TVM上执行
# ----------------------
# 过程与其他示例没有差别
from tvm.contrib import graph_runtime

m = graph_runtime.create(graph, lib, ctx)

# 设置输入
m.set_input('data', tvm.nd.array(data.astype(dtype)))
m.set_input(**params)
# 执行
print("Running the test image...")

m.run()
# 获得输出
tvm_out = []
if MODEL_NAME == 'yolov2':
    layer_out = {}
    layer_out['type'] = 'Region'
    # Get the region layer attributes (n, out_c, out_h, out_w, classes, coords, background)
    layer_attr = m.get_output(2).asnumpy()
    layer_out['biases'] = m.get_output(1).asnumpy()
    out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                 layer_attr[2], layer_attr[3])
    layer_out['output'] = m.get_output(0).asnumpy().reshape(out_shape)
    layer_out['classes'] = layer_attr[4]
    layer_out['coords'] = layer_attr[5]
    layer_out['background'] = layer_attr[6]
    tvm_out.append(layer_out)

elif MODEL_NAME == 'yolov3':
    for i in range(3):
        layer_out = {}
        layer_out['type'] = 'Yolo'
        # 获取YOLO层属性 (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i*4+3).asnumpy()
        layer_out['biases'] = m.get_output(i*4+2).asnumpy()
        layer_out['mask'] = m.get_output(i*4+1).asnumpy()
        out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                     layer_attr[2], layer_attr[3])
        layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
        layer_out['classes'] = layer_attr[4]
        tvm_out.append(layer_out)

# 检测，并进行框选标记。
thresh = 0.5
nms_thresh = 0.45
img = tvm.relay.testing.darknet.load_image_color(img_path)
_, im_h, im_w = img.shape
dets = tvm.relay.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh,
                                                      1, tvm_out)
last_layer = net.layers[net.n - 1]
tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)

# ******timepoint2-end*******
end2 = datetime.datetime.now()
# ******timepoint2-end*******
# print(end2-start2)

# coco_name = 'coco.names'
# coco_url = REPO_URL + 'data/' + coco_name + '?raw=true'
# font_name = 'arial.ttf'
# font_url = REPO_URL + 'data/' + font_name + '?raw=true'
# coco_path = download_testdata(coco_url, coco_name, module='data')
# font_path = download_testdata(font_url, font_name, module='data')
coco_path = "/home/previous/tvm8/yolov3/darknet/coco.names"
font_path = "/home/previous/tvm8/yolov3/darknet/arial.ttf"
# ******timepoint3-start*******
start3 = datetime.datetime.now()
# ******timepoint3-start*******

with open(coco_path) as f:
    content = f.readlines()

names = [x.strip() for x in content]

tvm.relay.testing.yolo_detection.draw_detections(font_path, img, dets, thresh, names, last_layer.classes)
# ******timepoint3-end*******
end3 = datetime.datetime.now()
# ******timepoint3-end*******

print(end1-start1)
print(end2-start2)
print(end3-start3)

plt.imshow(img.transpose(1, 2, 0))
plt.show()
