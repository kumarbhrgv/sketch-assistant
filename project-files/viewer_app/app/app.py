from flask import Flask,request
from keras.models import load_model
import ast
import  keras
import numpy as np
from PIL import Image
count = 0

app = Flask(__name__)


def process_text(req):
    res = [0,0,0,0,0,0,0,0]
    if 'bird' in req:
        res[6] = 1
        res[0] = 1
        res[1] = 1
        res[2] = 1
        res[3] = 1
        res[4] = 1
    if 'wing' in req:
        res[2] = 1
    if 'leg' in req:
        res[5] = 1
    if 'body' in req:
        res[4] = 1
    if 'eye' in req:
        res[7] = 1
    if 'beak' in req:
        res[1] = 1
    if 'head' in req:
        res[0] = 1
    print(res)
    return np.array(res).reshape(1,8)


@app.route("/")
def generate_image_from_text():
    generator_dir = "generator1000.h5"
    model = load_model(generator_dir)
    print('loaded model')
    req = request.args.get('text', None)
    text = process_text(req)
    noise = np.random.uniform(0, 1, (1, 8))
    print(noise.shape, text.shape)
    generated_images = model.predict([text, noise], verbose=0)
    generated_image = generated_images[0]
    generated_image = generated_image * 127.5 + 127.5
    generated_image = generated_image.reshape(128, 128)
    im = Image.fromarray(generated_image.astype(np.uint8))
    global count
    filename = 'gen' + str(count) + '.png'
    count+=1
    im.save(filename)
    return filename;


if __name__ == '__main__':
    app.run(debug=True)
    #generate_image_from_text(np.array([1,1,1,1,1,1,1,1]).reshape(1,8),model).show()
