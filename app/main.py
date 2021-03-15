from flask import Flask, jsonify, request, render_template
app = Flask(__name__)

from app.torch_utils import transform_image, get_prediction

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    prediction = get_prediction(tensor)
    data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)