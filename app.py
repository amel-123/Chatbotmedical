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

# Dictionaries
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

# ------------------ Symptom synonyms ------------------
symptom_synonyms = {
    "stomach ache":"stomach_pain", "belly pain":"stomach_pain", "tummy pain":"stomach_pain",
    "loose motion":"diarrhea", "motions":"diarrhea",
    "high temperature":"mild_fever", "temperature":"mild_fever", "feaver":"mild_fever",
    "coughing":"cough", "throat pain":"sore_throat",
    "cold":"chills", "breathing issue":"breathlessness", "shortness of breath":"breathlessness",
    "body ache":"muscle_pain", "fatigue":"fatigue", "tired":"fatigue", "exhausted":"fatigue",
    "thirsty":"polyuria",

    "very thirsty":"polyuria",
    "excessive_thirst":"polyuria",
    "hungry":"excessive_hunger",
    "frequent urination":"frequent_urination",
    "weight loss":"unexplained_weight_loss",
    "blurred vision":"blurred_vision",
    "slow healing":"slow_healing",
    "frequent infections":"frequent_infections"
}

# ------------------ Symptom extraction ------------------
def extract_symptoms(user_input, all_symptoms):
    extracted = []

    # 1️⃣ Nettoyage : minuscules, ponctuation → espace
    text = re.sub(r"[^\w\s]", " ", user_input.lower())
    text = text.replace("-", " ")

    # 2️⃣ Tokenisation en mots et phrases
    words = text.split()

    # 3️⃣ Vérifier les synonymes et mots exacts
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text and mapped not in extracted:
            extracted.append(mapped)

    # 4️⃣ Vérifier si un mot exact correspond à un symptôme
    for symptom in all_symptoms:
        sym_name = symptom.replace("_", " ")
        if sym_name in text and symptom not in extracted:
            extracted.append(symptom)

    # 5️⃣ Vérifier close match pour les mots individuels
    for word in words:
        if word in [s.replace("_", " ") for s in extracted]:
            continue
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.85)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0] and sym not in extracted:
                    extracted.append(sym)

    return list(set(extracted))



# ------------------ Prediction ------------------
def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class]*100,2)
    return disease, confidence, pred_proba

quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]

# ------------------ Routes ------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot')
def chatbot_page():
    session.clear()
    session['step'] = 'welcome'
    return render_template('index.html')


# Page IMC


# Médecins par catégorie d'IMC
imc_doctors = {
    "maigre": [
        {"nom": "Nutritionniste Salsabil Ben Arfa", "adresse": "Mutuelleville, 22 Rue Oum El Banine, Tunis, Tunisia", "contact": "50 330 220"},
    ],
    "normal": [],
    "surpoids": [
        {"nom": "Taieb Ben Alaya Nutritionniste -Diététicien", "adresse": "Rue Habib Chatti, Tunis 2092, Tunisie", "contact": "71 874 755"},
    ],
    "obese": [
               {"nom": "Taieb Ben Alaya Nutritionniste -Diététicien", "adresse": "Rue Habib Chatti, Tunis 2092, Tunisie", "contact": "71 874 755"},

    ]
}

@app.route('/imc', methods=['GET', 'POST'])
def imc_page():
    result = None
    doctors_list = []
    if request.method == 'POST':
        try:
            poids = float(request.form['poids'])
            taille = float(request.form['taille']) / 100  # convertir cm → m
            imc_val = poids / (taille**2)

            # Déterminer la catégorie
            if imc_val < 18.5:
                category = "maigre"
                result = f"Votre IMC est {imc_val:.2f} → Maigre"
            elif imc_val < 25:
                category = "normal"
                result = f"Votre IMC est {imc_val:.2f} → Normal"
            elif imc_val < 30:
                category = "surpoids"
                result = f"Votre IMC est {imc_val:.2f} → Surpoids"
            else:
                category = "obese"
                result = f"Votre IMC est {imc_val:.2f} → Obèse"

            # Récupérer la liste des médecins si disponible
            doctors_list = imc_doctors.get(category, [])
            
            #ajouter
            for doc in doctors_list:
                 doc["qr"] = generate_qr_code(doc)


        except:
            result = "Entrée invalide !"

    return render_template('imc.html', result=result, doctors_list=doctors_list)

def generate_qr_code(doctor):
    # 1. Nettoyage strict des données
    nom = str(doctor.get('nom', 'Medecin')).strip()
    tel = str(doctor.get('contact', '')).replace(' ', '').strip()
    adr = str(doctor.get('adresse', '')).strip()

    # 2. Format vCard 3.0 avec double champ Nom pour la compatibilité
    # N:Nom;Prenom;;; est souvent requis par Android
    # FN:Nom Complet est requis par iOS
    vcard_content = (
        "BEGIN:VCARD\n"
        "VERSION:3.0\n"
        f"N:;{nom};;;\n"
        f"FN:{nom}\n"
        f"TEL;TYPE=CELL:{tel}\n"
        f"ADR;TYPE=WORK:;;{adr}\n"
        "END:VCARD"
    )

    # 3. Création du nom de fichier unique
    clean_filename = "".join(x for x in nom if x.isalnum()) + ".png"
    folder = os.path.join(app.static_folder, "qrcodes")
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, clean_filename)

    # 4. Suppression de l'ancien fichier pour forcer la mise à jour
    if os.path.exists(path):
        os.remove(path)

    # 5. Génération du QR Code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(vcard_content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(path)

    return clean_filename


# ------------------ Chat logic ------------------
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json['message']
    step = session.get('step', 'welcome')

    if step == 'welcome':
        session['step'] = 'name'
        return jsonify(reply="🤖 Welcome to HealthCare ChatBot!\n👉 What is your name?")
    elif step == 'name':
        session['name'] = user_msg
        session['step'] = 'age'
        return jsonify(reply="👉 Please enter your age:")
    elif step == 'age':
        session['age'] = user_msg
        session['step'] = 'gender'
        return jsonify(reply="👉 What is your gender? (M/F):")
    elif step == 'gender':
        session['gender'] = user_msg
        session['step'] = 'symptoms'
        return jsonify(reply="👉 Describe your symptoms in a sentence:")
    elif step == 'symptoms':
        symptoms_list = extract_symptoms(user_msg, cols)
        if not symptoms_list:
            return jsonify(reply="❌ Could not detect valid symptoms. Please describe again:")
        session['symptoms'] = symptoms_list
        disease, conf, _ = predict_disease(symptoms_list)
        session['pred_disease'] = disease
        session['step'] = 'days'
        return jsonify(reply=f"✅ Detected symptoms: {', '.join(symptoms_list)}\n👉 For how many days have you had these symptoms?")
    elif step == 'days':
        session['days'] = user_msg
        session['step'] = 'severity'
        return jsonify(reply="👉 On a scale of 1–10, how severe is your condition?")
    elif step == 'severity':
        session['severity'] = user_msg
        session['step'] = 'preexist'
        return jsonify(reply="👉 Do you have any pre-existing conditions?")
    elif step == 'preexist':
        session['preexist'] = user_msg
        session['step'] = 'lifestyle'
        return jsonify(reply="👉 Do you smoke, drink alcohol, or have irregular sleep?")
    elif step == 'lifestyle':
        session['lifestyle'] = user_msg
        session['step'] = 'family'
        return jsonify(reply="👉 Any family history of similar illness?")
    elif step == 'family':
        session['family'] = user_msg
        disease = session['pred_disease']
        disease_symptoms = list(training[training['prognosis'] == disease].iloc[0][:-1].index[
            training[training['prognosis'] == disease].iloc[0][:-1] == 1
        ])
        session['disease_syms'] = disease_symptoms
        session['ask_index'] = 0
        session['step'] = 'guided'
        return ask_next_symptom()
    elif step == 'guided':
        idx = session['ask_index'] - 1
        if idx >= 0 and idx < len(session['disease_syms']):
            if user_msg.strip().lower() == 'yes':
                symptom = session['disease_syms'][idx]
                if symptom not in session['symptoms']:
                    session['symptoms'].append(symptom)
        return ask_next_symptom()
    elif step == 'final':
        return final_prediction()

def ask_next_symptom():
    i = session['ask_index']
    ds = session['disease_syms']
    while i < min(8, len(ds)):
        sym = ds[i]
        session['ask_index'] += 1
        # Skip if already in detected symptoms
        if sym in session['symptoms']:
            i += 1
            continue
        return jsonify(reply=f"👉 Do you also have {sym.replace('_',' ')}? (yes/no):")
    session['step'] = 'final'
    return final_prediction()

def final_prediction():
    disease, conf, _ = predict_disease(session['symptoms'])
    about = description_list.get(disease, 'No description available.')
    precautions = precautionDictionary.get(disease, [])
    text = (f"                        Result                            \n"
            f"\n🩺 Based on your answers, you may have **{disease}**\n"
            f"\n🔎 Confidence: {conf}%\n📖 About: {about}\n")
    if precautions:
        text += "\n\n🛡️ Suggested precautions:\n" + "\n\n".join(f"{i+1}. {p}" for i,p in enumerate(precautions))
    text += "\n\n\n💡 " + random.choice(quotes)
    text += f"\n\n\nThank you for using the chatbot. Wishing you good health, {session.get('name','User')}!"
    return jsonify(reply=text)

if __name__ == "__main__":
    app.run(debug=True)
