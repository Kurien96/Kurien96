from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
from math import radians, cos, sin, asin, sqrt

app = Flask(__name__)

model = load_model("waste_model.h5")

# ---------------- WASTE CENTERS (INCLUDING BMC SASTHAMCOTTA) ----------------
waste_centers = [
    {
        "name": "BMC College of Engineering, Sasthamcotta",
        "lat": 9.0360,
        "lon": 76.6370,
        "type": "General Waste, Paper, Plastic"
    },
    {
        "name": "Sasthamcotta Municipal Recycling Center",
        "lat": 9.0385,
        "lon": 76.6352,
        "type": "Plastic, Paper, Glass"
    },
    {
        "name": "E-Waste Collection Hub, Kollam",
        "lat": 8.8932,
        "lon": 76.6141,
        "type": "Battery, Electronics"
    },
    {
        "name": "Organic Compost Facility",
        "lat": 9.0328,
        "lon": 76.6409,
        "type": "Biological Waste"
    }
]

# ---------------- CLASSES ----------------
class_names = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# ---------------- DISPOSAL GUIDE ----------------
disposal_guide = {
    'battery': "❌ Never throw in normal trash\n✅ Drop at battery collection points or e-waste centers\n🔥 Tape battery terminals before disposal",
    'biological': "Dispose in biodegradable waste bin or compost.\nWhy: Prevents disease & supports composting.",
    'cardboard': "Recycle in paper/cardboard bin.\nKeep clean & dry.",
    'clothes': "Donate or recycle textiles.",
    'glass': "Wrap carefully and recycle.\nWhy: 100% recyclable.",
    'metal': "Send to metal recycling facility.\nWhy: Saves energy.",
    'paper': "Recycle in dry waste bin.\nWhy: Reduces deforestation.",
    'plastic': "Clean and recycle plastic waste.\nWhy: Prevents pollution.",
    'shoes': "Donate or dispose responsibly.",
    'trash': "Dispose in general waste bin.\nReduce as much as possible."
}

eco_impact = {
    'battery': "☠️ Prevents soil pollution",
    'biological': "🌱 Improves soil health",
    'cardboard': "📦 Highly recyclable",
    'clothes': "👕 Reduces textile waste",
    'glass': "🔄 100% recyclable",
    'metal': "⚡ Saves energy",
    'paper': "🌳 Saves trees",
    'plastic': "♻️ Saves ~1.5kg CO₂",
    'shoes': "👟 Reuse reduces landfill",
    'trash': "🚯 Reduce landfill waste"
}

location_rules = {
    "default": "Follow standard municipal waste rules.",
    "kerala": "Strict waste segregation is mandatory.",
    "bangalore": "Wet & dry waste separation required.",
    "chennai": "E-waste must go to authorized centers."
}

hazardous_waste = ['battery']

# ---------------- DISTANCE FUNCTION ----------------
def calculate_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return round(km, 2)

# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    data = {}

    if request.method == "POST":

        if "feedback" in request.form:
            with open("feedback.txt", "a") as f:
                f.write(request.form["feedback"] + "\n")

        file = request.files.get("image")
        location = request.form.get("location", "default")

        if file and file.filename:
            path = os.path.join("static", file.filename)
            file.save(path)

            img = image.load_img(path, target_size=(224, 224))
            img_arr = image.img_to_array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)

            preds = model.predict(img_arr)
            conf = round(float(np.max(preds)) * 100, 2)
            idx = np.argmax(preds)
            waste = class_names[idx]

            data = {
                "image": path,
                "waste": waste,
                "confidence": conf,
                "guide": disposal_guide[waste],
                "eco": eco_impact[waste],
                "location": location_rules[location],
                "hazard": waste in hazardous_waste,
                "low_conf": conf < 60
            }

    return render_template("index.html", data=data)

# ---------------- CAMERA ROUTE ----------------
@app.route("/camera", methods=["POST"])
def camera_predict():
    data = request.json["image"]
    img_bytes = base64.b64decode(data.split(",")[1])

    img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr)
    conf = round(float(np.max(preds)) * 100, 2)
    idx = np.argmax(preds)
    waste = class_names[idx]

    return jsonify({
        "waste": waste,
        "confidence": conf,
        "disposal": disposal_guide[waste],
        "eco": eco_impact[waste],
        "hazard": waste in hazardous_waste,
        "low_conf": conf < 60
    })

# ---------------- NEAREST WASTE CENTER ROUTE ----------------
@app.route("/nearest-center", methods=["POST"])
def nearest_center():
    user_lat = request.json["lat"]
    user_lon = request.json["lon"]

    nearest = None
    min_dist = float("inf")

    for center in waste_centers:
        dist = calculate_distance(
            user_lat, user_lon,
            center["lat"], center["lon"]
        )
        if dist < min_dist:
            min_dist = dist
            nearest = center

    return jsonify({
        "name": nearest["name"],
        "distance": min_dist,
        "type": nearest["type"]
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
