# app.py
from flask import Flask, render_template, request
import joblib
import os

# Tạo Flask app
app = Flask(__name__)

# Load mô hình đã huấn luyện
model_path = "svc_model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

def label_to_text(label):
    if label == 0:
        return "Tiêu cực"
    elif label == 1:
        return "Tích cực"
    else:
        return "Vui lòng nhập văn bản"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    text_input = ""
    
    if request.method == "POST":
        text_input = request.form.get("text")
        if text_input and model:
            prediction = model.predict([text_input])[0]

    return render_template("index.html", prediction=label_to_text(prediction), text_input=text_input)

if __name__ == "__main__":
    app.run(debug=True)
