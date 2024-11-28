from email_utils import send_email_with_attachment

# Paramètres pour le test
subject = "Test - Alerte Sécurité"
body = "Ceci est un e-mail de test envoyé depuis le script Python."
to_email = "votre_adresse_email_destinataire@gmail.com"
attachment_path = "./test_attachment.jpg"  # Assurez-vous que ce fichier existe dans le répertoire

# Exécution du test
try:
    send_email_with_attachment(subject, body, to_email, attachment_path)
    print("E-mail envoyé avec succès !")
except Exception as e:
    print(f"Erreur lors de l'envoi de l'e-mail : {e}")
