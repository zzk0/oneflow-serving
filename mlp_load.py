import numpy as np
import google.protobuf.text_format as text_format

import oneflow as flow
import oneflow.core.serving.saved_model_pb2 as saved_model_pb

from PIL import Image

def load_image(file):
    im = Image.open(file).convert("L")
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im - 128.0) / 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im


if __name__ == '__main__':
    sess = flow.serving.InferenceSession()
    sess.load_saved_model(saved_model_dir="./models")
    sess.launch()
    logits = sess.run(
        "mlp_inference",
        Input_14=load_image("./7.png"),
        Input_15=np.zeros((1,)).astype(np.int32))

    prediction = np.argmax(logits[0], 1)
    print("prediction: {}".format(prediction[0]))
    sess.close()
