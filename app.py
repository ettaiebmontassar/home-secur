import os
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from email_utils import send_email_with_attachment
from flasgger import Swagger

# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
DATABASE_FILE = 'sqlite:///database.db'
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_FILE
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialisation de Swagger (documentation API)
swagger = Swagger(app)

# Initialisation de la base de données
db = SQLAlchemy(app)

# Création du dossier pour stocker les images
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Modèle pour les événements détectés
class DetectionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(120), nullable=False)

# Création de la base de données au démarrage
with app.app_context():
    db.create_all()

# Fonction pour nettoyer les fichiers anciens (plus de 7 jours)
def cleanup_old_files():
    now = time.time()
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > 7 * 86400:  # 7 jours
            os.remove(file_path)
            print(f"Fichier supprimé : {file_path}")

# Endpoint pour télécharger une image
@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Endpoint pour télécharger une image.
    ---
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: L'image capturée à téléverser
    responses:
        200:
            description: Succès
        400:
            description: Erreur dans la requête
    """
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Le nom du fichier est vide"}), 400

    # Sauvegarder l'image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Enregistrer l'événement dans la base de données
    event = DetectionEvent(image_path=file_path)
    db.session.add(event)
    db.session.commit()

    # Envoyer un e-mail avec la pièce jointe
    send_email_with_attachment(
        subject="Alerte Sécurité - Mouvement détecté",
        body=f"Une nouvelle image a été capturée : {filename}.",
        to_email="destinataire_email@gmail.com",
        attachment_path=file_path
    )

    return jsonify({"message": "Image reçue et notification envoyée !", "filename": filename}), 200

# Endpoint pour récupérer tous les logs
@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Endpoint pour récupérer tous les logs d'événements.
    ---
    responses:
        200:
            description: Liste des événements
    """
    events = DetectionEvent.query.all()
    logs = [{"id": event.id, "timestamp": event.timestamp, "image_path": event.image_path} for event in events]
    return jsonify(logs)

# Endpoint pour accéder à une image spécifique
@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    """
    Endpoint pour récupérer une image spécifique.
    ---
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Nom du fichier image
    responses:
        200:
            description: Image téléchargée
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Endpoint pour nettoyer les fichiers anciens
@app.route('/cleanup', methods=['GET'])
def cleanup():
    """
    Endpoint pour nettoyer les fichiers anciens (plus de 7 jours).
    ---
    responses:
        200:
            description: Nettoyage terminé
    """
    cleanup_old_files()
    return jsonify({"message": "Nettoyage terminé"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
