import os
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

# Configuration des dossiers et Flask
KNOWN_FACES_DIR = "./known_faces"
ANNOTATED_IMAGES_DIR = "./annotated_images"
UPLOAD_FOLDER = './uploads'
DATABASE_FILE = 'sqlite:///database.db'

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

# Fonction d'entraînement
def train_model():
    logger.info("Entraînement du modèle LBPH...")
    # Simplifié pour les besoins de démonstration
    return {"example_label": "Example"}

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

# Endpoint de test pour l'envoi d'e-mails
@app.route('/test_email', methods=['GET'])
def test_email():
    try:
        test_image_path = os.path.join(ANNOTATED_IMAGES_DIR, "test_image.jpg")
        if not os.path.exists(test_image_path):
            logger.warning("Aucune image de test trouvée. Génération d'une image factice...")
            cv2.imwrite(test_image_path, np.zeros((100, 100, 3), dtype=np.uint8))

        send_alert_email(test_image_path)
        return jsonify({"message": "E-mail envoyé avec succès."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint pour entraîner le modèle
@app.route('/train', methods=['GET'])
def train_endpoint():
    global label_map
    try:
        label_map = train_model()
        return jsonify({"message": "Modèle entraîné avec succès."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        label_map = train_model()
        logger.info("Application Flask démarrée avec succès.")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Erreur critique au démarrage : {e}")
        exit(1)
