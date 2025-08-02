from flask import Flask, request, render_template
import numpy as np
import pickle

import json

# Load crop info
with open('crop_info.json',encoding='UTF-8') as f:
    crop_info = json.load(f)
# Load pre-trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract and convert input values to float
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Prepare feature vector
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply scaling
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)

        # Predict
        prediction = model.predict(sc_mx_features)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            info = crop_info.get(crop.lower(), {})
            return render_template("index.html",
                                   crop_name=crop,
                                   result="This crop grows well in your region!",
                                   image=info.get("image", ""),
                                   info=info)
        else:
            return render_template('index.html', result="Sorry, crop couldn't be identified.")

    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
