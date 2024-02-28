## Turn model to .h5

Github: keras-YOLOv3-model-set

input:
wget ......

output:
yolov3-tiny.h5

## Quantized model and Eolvation

### Run in docker
file: src/vai_quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/examples/quantization/keras/vitis/mnist_cnn_ptq.py


dataset path: qua_data

```
python3 quantized_model.py
```

input:
yolov3-tiny.h5

output:
yolov3-tiny-quantized.h5

## Compile model

### Run in docker

vai_c_tensorflow2

json find in `/opt/vitis_ai/compiler/arch` in docker tag: `xilinx/vitis-ai-tensorflow2-gpu   3.5.0.001-`

command:
```
vai_c_tensorflow2 -m yolov3-tiny-quantized.h5 -a arch.json --options '{"input_shape": "1,224,224,3"}' -o yolov3-tiny
vai_c_tensorflow2 -m new-yolov3-tiny-quantized.h5 -a arch.json --options '{"input_shape": "1,416,416,3"}' -o new-yolov3-tiny
```

output:
xmodel


env XLNX_ENABLE_DUMP=1 XLNX_ENABLE_DEBUG_MODE=1 XLNX_GOLDEN_DIR=./src/vai_quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/examples/quantization/keras/vitis/dump_results/dump_results_0 \
   xdputil run ./deploy.xmodel \
   ./hi.bin \
   2>result.log 1>&2

## Deploy

config: `/opt/xilinx/kv260-smartcam/share/vvas/`

- preprocess.json
- aiinference.json
- drawresult.json


xmodel: `/opt/xilinx/kv260-smartcam/share/vitis_ai_library/models/yolov3_coco_416_tf2/`

- xmodel
- label.json
- prototxt


### Run smartcam in docker

```
smartcam --mipi -W 1920 -H 1080 --target rtsp -a yolov3_coco
```

# TODO

1. 測試yolov3 tiny + 原教學的prototxt
2. yolov3的權重自己轉成xmodel
3. yolov3自行訓練