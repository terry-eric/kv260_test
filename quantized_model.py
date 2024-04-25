import tensorflow as tf

def load_and_preprocess_image(path):
    # 讀取圖片
    image = tf.io.read_file(path)
    # 解碼圖片
    image = tf.image.decode_jpeg(image, channels=3)
    # 調整圖片大小至模型所需的尺寸，例如YOLOv3 Tiny的輸入尺寸
    image = tf.image.resize(image, [416, 416])
    # 歸一化圖片
    image /= 255.0
    return image

# data_dir = "qua_data"
data_dir = "printing_qua_data"

# 創建一個從文件路徑數據集
dataset_images = tf.data.Dataset.list_files(f"{data_dir}/images/*.jpg")  # 假設所有圖片都是JPG格式
# 加載和預處理圖片
dataset_images = dataset_images.map(load_and_preprocess_image)
# 批處理，這裡的批大小根據你的需要和記憶體大小決定
dataset_images = dataset_images.batch(32)  # 例如，每批處理32張圖片

# 現在 dataset_images 可用於模型量化


from tensorflow_model_optimization.quantization.keras.vitis.layers import vitis_activation

model_name = "yolov3-printing"
model = tf.keras.models.load_model(f'{model_name}.h5')
model.summary()

# Post-Training Quantize
from tensorflow_model_optimization.quantization.keras import vitis_quantize

quantizer = vitis_quantize.VitisQuantizer(model)
# Quantized
quantized_model = quantizer.quantize_model(
    calib_dataset=dataset_images,
    include_cle=True,
    cle_steps=10,
    include_fast_ft=True)

quantized_model.save(f'{model_name}-quantized.h5')