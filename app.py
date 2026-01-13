from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import re, random, pandas as pd, numpy as np, csv, warnings
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import qrcode
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
app = Flask(__name__)
app.secret_key = "supersecret"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# ------------------ Load Data ------------------
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns  = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing  = testing.loc[:, ~testing.columns.duplicated()]
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)


severityDictionary, description_list, precautionDictionary = {}, {}, {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

def getDescription():
    with open('MasterData/symptom_Description.csv') as csv_file:
        for row in csv.reader(csv_file):
            description_list[row[0]] = row[1]

def getSeverityDict():
    with open('MasterData/symptom_severity.csv') as csv_file:
        for row in csv.reader(csv_file):
            try: severityDictionary[row[0]] = int(row[1])
            except: pass

def getprecautionDict():
    with open('MasterData/symptom_precaution.csv') as csv_file:
        for row in csv.reader(csv_file):
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

getSeverityDict(); getDescription(); getprecautionDict()

symptom_synonyms = {
    "stomach ache":"stomach_pain", "belly pain":"stomach_pain", "tummy pain":"stomach_pain",
    "loose motion":"diarrhea", "motions":"diarrhea",
    "high temperature":"mild_fever", "temperature":"mild_fever", "feaver":"mild_fever",
    "coughing":"cough", "throat pain":"sore_throat",
    "cold":"chills", "breathing issue":"breathlessness", "shortness of breath":"breathlessness",
    "body ache":"muscle_pain", "fatigue":"fatigue", "tired":"fatigue", "exhausted":"fatigue",
    "thirsty":"polyuria"
}

# ------------------ Extraction & Prediction ------------------
def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = re.sub(r"[^\w\s]", " ", user_input.lower())
    words = text.split()
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text and mapped not in extracted:
            extracted.append(mapped)
    for symptom in all_symptoms:
        sym_name = symptom.replace("_", " ")
        if sym_name in text and symptom not in extracted:
            extracted.append(symptom)
    for word in words:
        if word in [s.replace("_", " ") for s in extracted]: continue
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.85)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0] and sym not in extracted:
                    extracted.append(sym)
    return list(set(extracted))

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class]*100,2)
    return disease, confidence

quotes = ["🌸 Health is wealth.", "💪 A healthy outside starts from the inside.", "🌿 Self-care is not selfish."]

# ------------------ Routes IMC & QR ------------------
imc_doctors = {
    "underweight": [{"nom": "Nutritionniste Salsabil Ben Arfa", "adresse": "Mutuelleville, Tunis", "contact": "50 330 220"}],
    "normal": [],
    "overweight": [{"nom": "Taieb Ben Alaya Nutritionniste", "adresse": "Rue Habib Chatti, Tunis", "contact": "71 874 755"}],
    "obese": [{"nom": "Taieb Ben Alaya Nutritionniste", "adresse": "Rue Habib Chatti, Tunis", "contact": "71 874 755"}]
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/imc', methods=['GET', 'POST'])
def imc_page():
    result = None
    doctors_list = []
    if request.method == 'POST':
        try:
            poids = float(request.form['poids'])
            taille = float(request.form['taille'])
            
            # --- Controle de saisie Réel ---
            # Poids entre 2kg et 500kg | Taille entre 50cm et 250cm
            if not (2 <= poids <= 500) or not (50 <= taille <= 250):
                result = "Error: Please enter realistic values (Weight: 2-500kg, Height: 50-250cm)."
            else:
                taille_m = taille / 100
                imc_val = poids / (taille_m**2)
                
                if imc_val < 18.5: category = "underweight"
                elif imc_val < 25: category = "normal"
                elif imc_val < 30: category = "overweight"
                else: category = "obese"
                
                result = f"Your BMI is {imc_val:.2f} ({category.capitalize()})"
                doctors_list = imc_doctors.get(category, [])
                for doc in doctors_list:
                     doc["qr"] = generate_qr_code(doc)
        except ValueError:
            result = "Invalid input! Please enter numbers."
            
    return render_template('imc.html', result=result, doctors_list=doctors_list)

def generate_qr_code(doctor):
    nom = str(doctor.get('nom', 'Doctor')).strip()
    tel = str(doctor.get('contact', '')).replace(' ', '').strip()
    adr = str(doctor.get('adresse', '')).strip()
    vcard = f"BEGIN:VCARD\nVERSION:3.0\nFN:{nom}\nTEL;TYPE=CELL:{tel}\nADR;TYPE=WORK:;;{adr}\nEND:VCARD"
    
    clean_filename = "".join(x for x in nom if x.isalnum()) + ".png"
    folder = os.path.join(app.static_folder, "qrcodes")
    if not os.path.exists(folder): os.makedirs(folder)
    
    path = os.path.join(folder, clean_filename)
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(vcard)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(path)
    return clean_filename

# ------------------ Chatbot Logic (English) ------------------
@app.route('/chatbot')
def chatbot_page():
    session.clear()
    session['step'] = 'welcome'
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json['message'].strip().lower()
    step = session.get('step', 'welcome')

    if step == 'welcome':
        session['step'] = 'name'
        return jsonify(reply="🤖 Welcome to HealthCare Chatbot! What is your name?")

    elif step == 'name':
        # Name Validation (Letters only, min 2 chars)
        if not user_msg or not re.match(r"^[a-zA-Z\s]{2,}$", user_msg):
            return jsonify(reply="❌ Please enter a valid name (letters only, at least 2 characters):")
        session['name'] = user_msg.capitalize()
        session['step'] = 'symptoms'
        return jsonify(reply=f"Nice to meet you, {session['name']}. Please describe your symptoms in a sentence:")

    elif step == 'symptoms':
        symptoms_list = extract_symptoms(user_msg, cols)
        if not symptoms_list:
            return jsonify(reply="❌ I couldn't detect any symptoms. Could you please describe them differently?")
        
        session['symptoms'] = symptoms_list
        disease, _ = predict_disease(symptoms_list)
        session['pred_disease'] = disease
        
        # Prepare follow-up questions
        disease_symptoms = list(training[training['prognosis'] == disease].iloc[0][:-1].index[
            training[training['prognosis'] == disease].iloc[0][:-1] == 1
        ])
        session['disease_syms'] = disease_symptoms
        session['ask_index'] = 0
        session['step'] = 'guided'
        return ask_next_symptom()

    elif step == 'guided':
        # YES/NO Validation
        if user_msg not in ['yes', 'no', 'y', 'n']:
            return jsonify(reply="❌ Please answer with 'Yes' or 'No'.")

        idx = session['ask_index'] - 1
        if user_msg in ['yes', 'y']:
            sym = session['disease_syms'][idx]
            if sym not in session['symptoms']: session['symptoms'].append(sym)
        
        return ask_next_symptom()

    return jsonify(reply="Session error. Please restart the chat.")

def ask_next_symptom():
    i = session['ask_index']
    ds = session['disease_syms']
    while i < len(ds) and i < 8:
        sym = ds[i]
        session['ask_index'] += 1
        if sym in session['symptoms']:
            i += 1; continue
        return jsonify(reply=f"👉 Do you also experience: **{sym.replace('_',' ')}**? (yes/no)")
    
    session['step'] = 'final'
    return final_prediction()

def final_prediction():
    disease, conf = predict_disease(session['symptoms'])
    about = description_list.get(disease, 'No description available.')
    precautions = precautionDictionary.get(disease, [])
    
    text = (f"🩺 **Prediction: {disease}** ({conf}% match)\n\n"
            f"📖 **About:** {about}\n\n"
            f"🛡️ **Suggested Precautions:**\n" + 
            "".join(f"- {p}\n" for p in precautions if p.strip()))
    
    text += f"\n💡 {random.choice(quotes)}\nStay healthy, {session.get('name')}!"
    return jsonify(reply=text)

if __name__ == "__main__":
    app.run(debug=True)