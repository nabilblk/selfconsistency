from flask import Flask, Response, request, abort, jsonify, make_response
from flask_cors import CORS

import os
import uuid
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
import demo
import numpy as np
import cv2
import time
from pynvml import *
import io
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config.from_pyfile('config.cfg')

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
model_object = None


@app.route('/api/activate')
def activate():
    global model_object
    print('Activate Signal')

    if model_object is None:
        gpu_id = -1
        nvmlInit()
        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            memfree = meminfo.free / 1024.**2
            print('GPU_{} free Memory size: {} MB'.format(i,memfree))
            if memfree > 4300:
                gpu_id = i
        nvmlShutdown()
        ckpt_path = './ckpt/exif_final/exif_final.ckpt'
        model_object = demo.Demo(
            ckpt_path=ckpt_path, use_gpu=gpu_id, quality=3.0, num_per_dim=25)
        return jsonify({'status':'Model Loaded'})
    else:
        return jsonify({'status':'Model was already loaded'})


@app.route("/api/get_analysis", methods=["POST"])
def get_analysis():
    if model_object is None:
        return jsonify({'Error': 'Initialize the model first by calling /api/activate'})
    img = request.data
    # convert string of image data to uint8
    nparr = np.fromstring(img, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    ms_st = time.time()
    _, res1 = model_object(img, dense=True)  # Upsampled via bilinear upsampling
    print('MeanShift run time: %.3f' % (time.time() - ms_st))
    buf = io.BytesIO()
    plt.imsave(buf,1.0 - res1, cmap='jet', vmin=0.0, vmax=1.0,format='jpg')
    response = make_response(buf.getvalue())
    return response


@app.route('/api/desactivate')
def desactivate():
    global model_object
    del model_object
    model_object = None
    return jsonify({'status':'Model removed from main memory'})


if __name__ == '__main__':
    host = app.config['HOST']
    port = int(app.config['PORT'])
    app.run(host=host, port=port)
