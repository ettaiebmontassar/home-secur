import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
from dotenv import load_dotenv
import logging

# Initialisation des logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("FlaskApp")

# Charger les variables d'environnement
load_dotenv()

EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT')

# Dossiers pour les données
KNOWN_FACES_DIR = "./known_faces"
ANNOTATED_IMAGES_DIR = "./annotated_images"
UPLOAD_FOLDER = './uploads'
DATABASE_FILE = 'sqlite:///database.db'

# Initialisation de l'application Flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_FILE
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialisation de la base de données
db = SQLAlchemy(app)

# Création des dossiers nécessaires
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Modèle pour la base de données
class DetectionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(120), nullable=False)
    annotated_image_path = db.Column(db.String(120), nullable=True)

with app.app_context():
    db.create_all()

# Initialisation du modèle LBPH
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
label_map = None  # Variable pour mapper les labels à des noms

# Fonction pour entraîner le modèle
def train_model():
    logger.info("Entraînement du modèle LBPH...")
    faces = []
    labels = []
    label_map = {}
    label_id = 0
    IMAGE_SIZE = (200, 200)

    if not os.listdir(KNOWN_FACES_DIR):
        logger.error("Le dossier `known_faces` est vide.")
        raise ValueError("Ajoutez des images dans `known_faces` pour entraîner le modèle.")

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[label_id] = person_name

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            resized_image = cv2.resize(image, IMAGE_SIZE)
            faces.append(resized_image)
            labels.append(label_id)

        label_id += 1

    if not faces:
        raise ValueError("Aucun visage valide trouvé pour l'entraînement.")

    face_recognizer.train(np.array(faces), np.array(labels))
    return label_map

# Fonction pour envoyer un e-mail
def send_alert_email(image_path):
    try:
        logger.info("Préparation de l'e-mail...")
        subject = "⚠️ Alerte de sécurité : Visage inconnu détecté"
        body = "Un visage inconnu a été détecté. Veuillez vérifier l'image jointe."

        message = MIMEMultipart()
        message["From"] = EMAIL_SENDER
        message["To"] = EMAIL_RECIPIENT
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        with open(image_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(image_path)}"
        )
        message.attach(part)

        logger.info("Connexion au serveur SMTP...")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(message)
        logger.info("E-mail envoyé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'e-mail : {e}")

# Fonction pour analyser une image
def detect_and_recognize_faces(image_path, label_map):
    try:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            annotated_image_path = os.path.join(ANNOTATED_IMAGES_DIR, f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(annotated_image_path, image)
            return True, annotated_image_path
    except Exception as e:
        logger.error(f"Erreur pendant la détection : {e}")
        return False, None

# Endpoint pour téléverser et analyser une image
@app.route('/upload', methods=['POST'])
def upload_and_analyze_image():
    global label_map
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier envoyé."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné."}), 400

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    unknown_detected, annotated_image_path = detect_and_recognize_faces(file_path, label_map)

    new_event = DetectionEvent(
        image_path=f"/uploads/{filename}",
        annotated_image_path=f"/annotated_images/{os.path.basename(annotated_image_path)}"
    )
    db.session.add(new_event)
    db.session.commit()

    if unknown_detected:
        send_alert_email(annotated_image_path)

    return jsonify({"message": "Image analysée.", "unknown_detected": unknown_detected}), 200

# Endpoint pour afficher les événements
@app.route('/events', methods=['GET'])
def get_detection_events():
    events = DetectionEvent.query.all()
    events_data = [
        {
            "id": event.id,
            "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": event.image_path.replace("\\", "/"),
            "annotated_image_path": event.annotated_image_path.replace("\\", "/")
        }
        for event in events
    ]
    return jsonify({"events": events_data}), 200

# Ajout des routes pour servir les fichiers
@app.route('/uploads/<filename>', methods=['GET'])
def serve_uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/annotated_images/<filename>', methods=['GET'])
def serve_annotated_image(filename):
    return send_from_directory(ANNOTATED_IMAGES_DIR, filename)

if __name__ == '__main__':
    try:
        label_map = train_model()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Erreur critique au démarrage : {e}")
        exit(1)
