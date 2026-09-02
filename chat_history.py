"""Persistance de l'historique de conversation (fichier JSON sur disque).

Même philosophie que l'index ChromaDB : ce qui doit survivre aux
redémarrages et aux rafraîchissements de page vit sur disque, pas dans
st.session_state. Module sans Streamlit — c'est app.py qui décide quand
charger et sauvegarder.
"""

import json
from pathlib import Path

# Ancré au dossier du projet (pas au dossier courant).
_FILE = Path(__file__).resolve().parent / "chat_history.json"


def load():
    """Retourne l'historique sauvegardé, ou [] s'il n'y en a pas.

    Un fichier absent ou corrompu ne doit jamais faire planter l'app :
    on repart simplement d'une conversation vide.
    """
    try:
        return json.loads(_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save(messages):
    """Écrit l'historique complet (liste de dicts JSON-sérialisables)."""
    _FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2),
                     encoding="utf-8")


def clear():
    """Efface l'historique sauvegardé (bouton « Nouvelle conversation »)."""
    _FILE.unlink(missing_ok=True)
