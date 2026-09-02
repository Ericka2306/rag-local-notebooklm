"""Authentification multi-utilisateur (streamlit-authenticator).

Extension au-delà du sujet du TP : chaque utilisateur a son compte, et
ses données (index vectoriel, historique de conversation) sont cloisonnées
par identifiant — cf. rag/ingestion.py et chat_history.py.

Ce que le paquet streamlit-authenticator apporte :
  - vérification des mots de passe par hachage bcrypt (users.yaml ne
    contient jamais de mot de passe en clair) ;
  - un cookie signé qui maintient la session à travers les F5 et les
    redémarrages, jusqu'à expiration ou déconnexion.
"""

from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
import yaml

_CONFIG_FILE = Path(__file__).resolve().parent / "users.yaml"


def get_authenticator():
    """Construit l'authentificateur — à CHAQUE exécution du script.

    Surtout pas de @st.cache_resource ici : le constructeur crée le
    composant navigateur qui lit/écrit le cookie de session, et un widget
    ne doit jamais vivre dans une fonction cachée (il ne serait pas rejoué
    aux reruns suivants — cookie illisible, session cassée). La création
    est légère, la recréer à chaque rerun est l'usage documenté.
    """
    config = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8"))
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )


def require_login():
    """Affiche le formulaire de connexion et BLOQUE tant que non connecté.

    Retourne (authenticator, username) une fois l'utilisateur authentifié.
    Le st.stop() garantit qu'aucune donnée n'est affichée ni chargée sans
    session valide — c'est le portail d'entrée de l'application.
    """
    authenticator = get_authenticator()
    authenticator.login(
        location="main",
        fields={"Form name": "Connexion", "Username": "Identifiant",
                "Password": "Mot de passe", "Login": "Se connecter"},
    )

    status = st.session_state.get("authentication_status")
    if status is False:
        st.error("Identifiant ou mot de passe incorrect.")
        st.stop()
    if status is None:
        st.info("Connectez-vous pour accéder à vos documents "
                "(compte de démo : demo / demo1234).")
        st.stop()

    return authenticator, st.session_state["username"]
