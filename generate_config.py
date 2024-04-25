import json
import shutil
import sys
from pathlib import Path

COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (192, 192, 192),
    (128, 128, 128),
    (128, 0, 0),
    (128, 128, 0),
    (0, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (0, 0, 128),
)

with open("models/label.json") as f:
    label = json.load(f)

with open("vvas/drawresult.json") as f:
    drawresult = json.load(f)

with open("vvas/aiinference.json") as f:
    aiinference = json.load(f)

with open("models/yolov3_tiny.prototxt") as f:
    prototxt = f.read()


def generate(org_xmodel_name, label_name, input_size=""):
    input_size = str(input_size)
    with open(label_name) as f:
        labels = f.read().splitlines()

    xmodel_path = Path(org_xmodel_name)
    model_name = (
        xmodel_path.stem if input_size == "" else xmodel_path.stem + "_" + input_size
    )
    save_vvas_path = Path(f"{model_name}_vvas")
    save_vvas_path.mkdir(parents=True, exist_ok=True)
    save_model_path = Path(f"{model_name}_model")
    save_model_path.mkdir(parents=True, exist_ok=True)

    # aiinference.json
    aiinference["kernels"][0]["config"]["model-name"] = model_name

    # label.json
    label["model-name"] = model_name
    label["num-labels"] = len(labels)
    for index, name in enumerate(labels):
        label["labels"].append({"label": index, "name": name, "display_name": name})
        color = COLORS[index % len(COLORS)]
        drawresult["kernels"][0]["config"]["classes"].append(
            {"name": name, "blue": color[0], "green": color[1], "red": color[2]}
        )

    with open(f"{save_vvas_path}/aiinference.json", "w") as f:
        json.dump(aiinference, f, indent=4)
    with open(f"{save_vvas_path}/drawresult.json", "w") as f:
        json.dump(drawresult, f, indent=4)
    shutil.copy("vvas/preprocess.json", f"{save_vvas_path}/preprocess.json")

    # model
    new_prototxt = prototxt.replace("replace_num_classes", str(len(labels)))
    with open(f"{save_model_path}/{model_name}.prototxt", "w") as f:
        f.write(new_prototxt)
    shutil.copy(org_xmodel_name, f"{save_model_path}/{model_name}.xmodel")
    with open(f"{save_model_path}/label.json", "w") as f:
        json.dump(label, f, indent=4)

    print("model saved in:", save_model_path)
    print("vvas config saved in:", save_vvas_path)


if __name__ == "__main__":
    xmodel_name = "yolov3_printing.xmodel"
    label_name = "labels.txt"
    input_size = ""

    if len(sys.argv) > 1:
        xmodel_name = sys.argv[1]
    if len(sys.argv) > 2:
        label_name = sys.argv[2]
    if len(sys.argv) > 3:
        input_size = sys.argv[3]
    generate(xmodel_name, label_name, input_size)
