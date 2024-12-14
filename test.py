import os
import cv2
import numpy as np
import logging

# Configuration des logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("TestModel")

# Dossier contenant les visages connus
KNOWN_FACES_DIR = "./known_faces"
IMAGE_SIZE = (200, 200)

def test_known_faces_directory():
    """
    Vérifie que le dossier `known_faces` contient des sous-dossiers avec des images valides.
    """
    logger.info("Vérification du dossier `known_faces`...")
    if not os.path.exists(KNOWN_FACES_DIR):
        logger.error(f"Le dossier `{KNOWN_FACES_DIR}` n'existe pas.")
        return False

    if not os.listdir(KNOWN_FACES_DIR):
        logger.error("Le dossier `known_faces` est vide. Ajoutez des images avant de tester.")
        return False

    valid_images = True
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            logger.warning(f"Ignoré : {person_name} n'est pas un dossier.")
            continue

        logger.info(f"Vérification des images pour : {person_name}")
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.error(f"Erreur : Impossible de lire l'image `{image_path}`.")
                valid_images = False
            else:
                logger.info(f"Image `{image_name}` chargée avec succès.")

    return valid_images

def test_train_model():
    """
    Tente d'entraîner un modèle LBPH avec les images dans le dossier `known_faces`.
    """
    logger.info("Entraînement du modèle LBPH...")
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[label_id] = person_name
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Erreur : Impossible de charger {image_path}. Ignoré.")
                continue

            resized_image = cv2.resize(image, IMAGE_SIZE)
            faces.append(resized_image)
            labels.append(label_id)

        logger.info(f"Images pour `{person_name}` chargées avec succès.")
        label_id += 1

    if len(faces) == 0:
        logger.error("Erreur : Aucun visage chargé. L'entraînement ne peut pas continuer.")
        return False

    faces = np.array(faces, dtype="uint8")
    labels = np.array(labels, dtype="int32")

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, labels)
    logger.info("Modèle LBPH entraîné avec succès.")
    return True

if __name__ == "__main__":
    logger.info("===== Début du test =====")

    # Étape 1 : Vérification des données
    if not test_known_faces_directory():
        logger.error("Échec du test : Vérifiez le contenu du dossier `known_faces`.")
        exit(1)

    # Étape 2 : Entraînement du modèle
    if not test_train_model():
        logger.error("Échec du test : Entraînement du modèle impossible.")
        exit(1)

    logger.info("Tous les tests ont été passés avec succès.")
