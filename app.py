import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import logging

# Initialisation des logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("FlaskApp")

# Configuration Flask
KNOWN_FACES_DIR = "./known_faces"
ANNOTATED_IMAGES_DIR = "./annotated_images"
UPLOAD_FOLDER = './uploads'
DATABASE_FILE = 'sqlite:///database.db'

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_FILE
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_IMAGES_FOLDER'] = ANNOTATED_IMAGES_DIR

# Initialisation de la base de données
db = SQLAlchemy(app)

# Création des dossiers nécessaires
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Modèle LBPH
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
label_map = None  # Variable globale pour stocker le mapping des labels


# Modèle de base de données pour les événements
class DetectionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(120), nullable=False)
    annotated_image_path = db.Column(db.String(120), nullable=True)

# Création de la base de données
with app.app_context():
    db.create_all()


# Fonction d'entraînement du modèle LBPH
def train_model():
    logger.info("Entraînement du modèle LBPH...")
    faces = []
    labels = []
    label_map = {}
    label_id = 0
    IMAGE_SIZE = (200, 200)

    if not os.listdir(KNOWN_FACES_DIR):
        logger.error("Le dossier `known_faces` est vide. Ajoutez des images pour entraîner le modèle.")
        raise ValueError("Le dossier `known_faces` est vide.")

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[label_id] = person_name
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Erreur : Impossible de charger l'image `{image_path}`.")
                continue

            resized_image = cv2.resize(image, IMAGE_SIZE)
            faces.append(resized_image)
            labels.append(label_id)

        logger.info(f"Images pour `{person_name}` chargées avec succès.")
        label_id += 1

    if not faces:
        logger.error("Erreur : Aucun visage valide trouvé. Entraînement annulé.")
        raise ValueError("Aucun visage valide trouvé.")

    face_recognizer.train(np.array(faces), np.array(labels))
    logger.info("Modèle LBPH entraîné avec succès.")
    return label_map


# Endpoint pour l'upload et l'analyse d'une image
@app.route('/upload', methods=['POST'])
def upload_and_analyze_image():
    global label_map
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé. Utilisez le champ 'file'."}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné."}), 400

        # Sauvegarder le fichier
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Vérifier si le modèle est prêt
        if label_map is None:
            return jsonify({"error": "Le modèle n'a pas été entraîné. Ajoutez des visages connus."}), 500

        # Analyser l'image
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"message": "Aucun visage détecté."}), 200

        # Annoter l'image
        for (x, y, w, h) in faces:
            face = gray_image[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            label, confidence = face_recognizer.predict(face)
            name = label_map.get(label, "Inconnu") if confidence < 50 else "Inconnu"
            color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        annotated_path = os.path.join(app.config['ANNOTATED_IMAGES_FOLDER'], f"annotated_{timestamp}.jpg")
        cv2.imwrite(annotated_path, image)

        # Ajouter un événement à la base de données
        event = DetectionEvent(image_path=file_path, annotated_image_path=annotated_path)
        db.session.add(event)
        db.session.commit()

        return jsonify({
            "message": "Image analysée.",
            "annotated_image_path": annotated_path
        }), 200
    except Exception as e:
        logger.error(f"Erreur : {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500


# Endpoint de test pour valider la configuration
@app.route('/test', methods=['GET'])
def test_endpoint():
    try:
        # Vérifie la disponibilité des données et entraîne le modèle
        train_model()
        return jsonify({"message": "Tous les tests ont été passés avec succès."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    try:
        label_map = train_model()
    except Exception as e:
        logger.error(f"Erreur critique : {e}")
        exit(1)  # Arrêter si le modèle ne peut pas être entraîné
    app.run(debug=True, host='0.0.0.0', port=5000)
