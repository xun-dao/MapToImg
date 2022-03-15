# *_*coding:utf-8 *_*
"""
@author: mingruisu
@time: 2022/3/13 6:46 PM
@desc:
"""
import base64
import os

import cv2
import flask
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

from app.src.service import inpaint_edge_service
from app.src.service.config import Config

#
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
#
config_path = os.path.join(BASE_DIR, "MapToImg/app/src/checkpoints/celeba", 'config.yml')
print(config_path)
config = Config(config_path)
config.MODE = 2



app = Flask(
    __name__,
    template_folder="app/templates",
    static_url_path="/app/static",
    static_folder="app/static",
)
app.config["MAX_CONTENT_LENGTH"] = 10000000  # allow 10 MB post
# 加载模型
edge_model, inpaint_model = inpaint_edge_service.load_model(config)

@app.route("/")
def index():
    image_paths = []
    return render_template(
        "index.html",
        canvas_size=256,
        base_path=base_path,
        image_paths=list(os.listdir(base_path)),
    )


@app.route("/post", methods=["POST"])
def post():
    if request.method == "POST":

        mask_data = base64.b64decode(request.json["data_reference"][0])
        mask_img_data = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_COLOR)


        reference_img_name = request.json["references"][0]
        reference_data = cv2.imread(f"app/static/components/img/celeba_hq/{reference_img_name}")
        results_path = f"app/static/inpaint_path/"
        result_img_name = f"inpaint_{reference_img_name}"

        # 制作涂抹图片
        img_with_mask_path = f"app/static/images/with_mask_{reference_img_name}"
        print("mask_img_data", mask_img_data.shape)
        print("reference_data", reference_data.shape)
        res = cv2.addWeighted(reference_data, 1, mask_img_data, 1, 0)
        cv2.imwrite(img_with_mask_path, res)


        # 使用Numpy创建一张A4(2105×1487)纸
        black_img = np.zeros(mask_img_data.shape, np.uint8)
        black_img.fill(0)

        mask_save_path = f"app/static/mask/mask_{reference_img_name}"
        mask_res = cv2.addWeighted(black_img, 1, mask_img_data, 1, 0)
        cv2.imwrite(mask_save_path, mask_res)

        # 获取数据
        img, img_gray, edge, mask = inpaint_edge_service.get_data(0, img_with_mask_path, mask_save_path)

        # 执行推理
        forward_result_path = inpaint_edge_service.forward(img_gray.to(config.DEVICE), edge.to(config.DEVICE),
                                                           mask.to(config.DEVICE),
                                                           img.to(config.DEVICE), edge_model, inpaint_model,
                                                           results_path, result_img_name)
        print(forward_result_path)


        return flask.jsonify(result=[forward_result_path])
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    base_path = f"app/static/components/img/celeba_hq/"
    app.debug = True
    app.run(host="127.0.0.1", port=6006)