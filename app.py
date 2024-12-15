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
            logger.warning(f"Ignoré : {person_name} n'est pas un dossier.")
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

        label_id += 1

    if not faces:
        logger.error("Aucun visage valide trouvé pour l'entraînement.")
        raise ValueError("Entraînement annulé : aucun visage valide.")

    face_recognizer.train(np.array(faces), np.array(labels))
    logger.info(f"Modèle LBPH entraîné avec {len(faces)} visages et {len(label_map)} classes.")
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
        if image is None:
            logger.error("Image non valide ou introuvable.")
            return False, None

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            logger.info("Aucun visage détecté.")
            return False, None

        for (x, y, w, h) in faces:
            face = gray_image[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            label, confidence = face_recognizer.predict(face)
            name = label_map.get(label, "Inconnu") if confidence < 50 else "Inconnu"
            color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_image_path = os.path.join(ANNOTATED_IMAGES_DIR, f"annotated_{timestamp}.jpg")
        cv2.imwrite(annotated_image_path, image)

        return True, annotated_image_path
    except Exception as e:
        logger.error(f"Erreur pendant la détection : {e}")
        raise

# Endpoint pour téléverser et analyser une image
@app.route('/upload', methods=['POST'])
def upload_and_analyze_image():
    global label_map
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé."}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné."}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        logger.info(f"Fichier sauvegardé : {file_path}")

        if label_map is None:
            return jsonify({"error": "Le modèle n'a pas été entraîné."}), 500

        unknown_detected, annotated_image_path = detect_and_recognize_faces(file_path, label_map)
        if unknown_detected:
            send_alert_email(annotated_image_path)

        return jsonify({"message": "Image analysée.", "unknown_detected": unknown_detected}), 200
    except Exception as e:
        logger.error(f"Erreur pendant l'analyse : {e}")
        return jsonify({"error": "Erreur interne."}), 500

# Endpoint pour entraîner le modèle manuellement
@app.route('/train', methods=['GET'])
def train_model_endpoint():
    global label_map
    try:
        label_map = train_model()
        return jsonify({"message": "Modèle LBPH entraîné avec succès."}), 200
    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement : {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint pour afficher les événements
@app.route('/events', methods=['GET'])
def get_detection_events():
    try:
        events = DetectionEvent.query.all()
        events_data = [
            {
                "id": event.id,
                "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": event.image_path,
                "annotated_image_path": event.annotated_image_path
            }
            for event in events
        ]
        return jsonify({"events": events_data}), 200
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des événements : {e}")
        return jsonify({"error": "Erreur interne."}), 500

# Endpoint pour supprimer tous les événements
@app.route('/delete_all_events', methods=['DELETE'])
def delete_all_detection_events():
    try:
        num_deleted = DetectionEvent.query.delete()
        db.session.commit()
        logger.info(f"{num_deleted} événements supprimés.")
        return jsonify({"message": f"{num_deleted} événements supprimés avec succès."}), 200
    except Exception as e:
        logger.error(f"Erreur lors de la suppression des événements : {e}")
        return jsonify({"error": "Erreur interne."}), 500

# Endpoint pour tester l'envoi d'e-mails
@app.route('/test_email', methods=['GET'])
def test_email():
    try:
        test_image_path = os.path.join(ANNOTATED_IMAGES_DIR, "test_image.jpg")
        if not os.path.exists(test_image_path):
            logger.warning("Génération d'une image factice pour test...")
            cv2.imwrite(test_image_path, np.zeros((100, 100, 3), dtype=np.uint8))

        send_alert_email(test_image_path)
        return jsonify({"message": "E-mail envoyé avec succès."}), 200
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
