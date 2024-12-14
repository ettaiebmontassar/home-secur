import os
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
from dotenv import load_dotenv
import logging

# Initialiser les logs
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
app.config['ANNOTATED_IMAGES_FOLDER'] = ANNOTATED_IMAGES_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite : 16 Mo

# Initialisation de la base de données
db = SQLAlchemy(app)

# Création des dossiers nécessaires
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Modèle LBPH pour la reconnaissance faciale
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Variable globale pour stocker le label_map
label_map = None

# Modèle pour les événements détectés
class DetectionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(120), nullable=False)
    annotated_image_path = db.Column(db.String(120), nullable=True)

# Création de la base de données au démarrage
with app.app_context():
    db.create_all()


def send_alert_email(image_path):
    """
    Envoie un e-mail d'alerte avec l'image annotée en pièce jointe.
    """
    try:
        subject = "⚠️ Alerte de sécurité : Visage inconnu détecté"
        body = "Un visage inconnu a été détecté par le système de sécurité. Veuillez vérifier l'image en pièce jointe."

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

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(message)

        logger.info("Alerte email envoyée avec l'image annotée.")
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'email : {e}")


def train_model():
    """
    Entraîne le modèle LBPH avec les visages connus dans `KNOWN_FACES_DIR`.
    """
    logger.info("Entraînement du modèle LBPH...")
    faces = []
    labels = []
    label_map = {}
    label_id = 0
    IMAGE_SIZE = (200, 200)

    if not os.path.exists(KNOWN_FACES_DIR) or not os.listdir(KNOWN_FACES_DIR):
        logger.error("Le dossier `known_faces` est vide ou introuvable.")
        raise ValueError("Ajoutez des images dans `known_faces` pour entraîner le modèle.")

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            logger.warning(f"Ignoré : {person_name} n'est pas un dossier valide.")
            continue

        label_map[label_id] = person_name
        logger.info(f"Lecture des images pour : {person_name}")

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Image non valide ignorée : {image_path}")
                continue

            resized_image = cv2.resize(image, IMAGE_SIZE)
            faces.append(resized_image)
            labels.append(label_id)

        logger.info(f"{len(faces)} images chargées pour `{person_name}`.")
        label_id += 1

    if not faces:
        logger.error("Aucun visage valide trouvé pour entraîner le modèle.")
        raise ValueError("Entraînement annulé : aucun visage valide.")

    face_recognizer.train(np.array(faces), np.array(labels))
    logger.info("Modèle LBPH entraîné avec succès.")
    return label_map


@app.route('/upload', methods=['POST'])
def upload_and_analyze_image():
    """
    Téléverse et analyse une image envoyée pour détection et reconnaissance faciale.
    """
    global label_map
    logger.info("Requête reçue pour téléversement d'image.")
    try:
        if 'file' not in request.files:
            logger.error("Aucun fichier trouvé dans la requête.")
            return jsonify({"error": "Aucun fichier envoyé. Utilisez le champ 'file'."}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("Aucun fichier sélectionné.")
            return jsonify({"error": "Aucun fichier sélectionné."}), 400

        # Sauvegarder le fichier
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"Fichier sauvegardé : {file_path}")

        # Vérifier si le modèle est prêt
        if label_map is None:
            logger.error("Le modèle LBPH n'a pas été entraîné.")
            return jsonify({"error": "Le modèle n'a pas été entraîné. Ajoutez des visages connus."}), 500

        # Détection et reconnaissance
        unknown_detected, annotated_image_path = detect_and_recognize_faces(file_path, label_map)
        logger.info(f"Analyse terminée. Visage inconnu : {unknown_detected}")

        # Ajouter un événement à la base de données
        event = DetectionEvent(image_path=file_path, annotated_image_path=annotated_image_path)
        db.session.add(event)
        db.session.commit()
        logger.info("Événement enregistré dans la base de données.")

        return jsonify({
            "message": "Image reçue et analysée.",
            "filename": filename,
            "unknown_detected": unknown_detected
        }), 200
    except Exception as e:
        logger.error(f"Erreur pendant le traitement de l'image : {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500


if __name__ == '__main__':
    try:
        label_map = train_model()
        logger.info("Application démarrée avec succès.")
    except Exception as e:
        logger.error(f"Erreur critique : {e}")
        exit(1)
    app.run(debug=True, host='0.0.0.0', port=5000)
