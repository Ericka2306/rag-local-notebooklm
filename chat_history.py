"""Persistance de l'historique de conversation (fichier JSON sur disque).

Même philosophie que l'index ChromaDB : ce qui doit survivre aux
redémarrages et aux rafraîchissements de page vit sur disque, pas dans
st.session_state. Cloisonnement multi-utilisateur : un fichier par
identifiant (chat_history_<user>.json). Module sans Streamlit — c'est
app.py qui décide quand charger et sauvegarder.
"""

import json
from pathlib import Path

# Ancré au dossier du projet (pas au dossier courant).
_DIR = Path(__file__).resolve().parent


def _file(user):
    """Chemin du fichier d'historique de CET utilisateur."""
    return _DIR / f"chat_history_{user}.json"


def load(user):
    """Retourne l'historique sauvegardé de l'utilisateur, ou [].

    Un fichier absent ou corrompu ne doit jamais faire planter l'app :
    on repart simplement d'une conversation vide.
    """
    try:
        return json.loads(_file(user).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save(messages, user):
    """Écrit l'historique complet (liste de dicts JSON-sérialisables)."""
    _file(user).write_text(
        json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def clear(user):
    """Efface l'historique sauvegardé (bouton « Nouvelle conversation »)."""
    _file(user).unlink(missing_ok=True)
