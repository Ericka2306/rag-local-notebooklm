"""
Prépare l'archive de rendu en vérifiant les consignes du TP
============================================================
Les consignes pénalisent un JSON mal formé et interdisent tout fichier non
demandé. Ce script vérifie donc tout avant de construire l'archive :

  1. rendu/etape.json est un JSON valide, avec la structure du modèle
     fourni et tous les champs remplis ;
  2. les trois fichiers Python déclarés existent et compilent ;
  3. les deux captures d'écran sont présentes dans rendu/ ;
  4. l'archive <numéro>.zip contient UNIQUEMENT le dossier <numéro>/ avec
     etape.json, les trois fichiers .py et les deux captures.

Usage :  python preparer_rendu.py
"""

import json
import py_compile
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RENDU = ROOT / "rendu"
SCREENSHOTS = ("preuve_semantique.png", "preuve_rag_complet.png")
PY_KEYS = ("etape_1_interface", "etape_2_ingestion", "etape_3_4_rag")


def fail(message):
    """Affiche l'erreur et arrête le script."""
    sys.exit(f"✗ {message}")


def check_json():
    """Valide etape.json et renvoie son contenu."""
    path = RENDU / "etape.json"
    if not path.is_file():
        fail(f"{path} introuvable")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"etape.json n'est pas un JSON valide : {exc}")

    expected = {
        "etudiant": ("nom", "prenom", "numero_etudiant"),
        "fichiers": PY_KEYS,
        "justification_chunking": ("taille_chunk_fixe", "explication"),
        "environnement": ("modele_ollama", "modele_embedding"),
    }
    if set(data) != set(expected):
        fail(f"sections attendues : {sorted(expected)}, trouvées : {sorted(data)}")
    for section, keys in expected.items():
        if set(data[section]) != set(keys):
            fail(f"clés de '{section}' attendues : {sorted(keys)}")
        for key in keys:
            value = data[section][key]
            if key == "taille_chunk_fixe":
                if not isinstance(value, bool):
                    fail("taille_chunk_fixe doit valoir true ou false")
            elif not isinstance(value, str) or not value.strip():
                fail(f"champ vide : {section}.{key}")

    explanation = data["justification_chunking"]["explication"]
    if explanation.startswith("Rédigez ici"):
        fail("l'explication du chunking est encore le texte du modèle")
    sentences = len(re.findall(r"[.!?](?:\s|$)", explanation))
    if sentences > 7:
        fail(f"explication trop longue : {sentences} phrases (7 maximum)")

    print(f"✓ etape.json valide ({sentences} phrases de justification)")
    return data


def check_python_files(data):
    """Vérifie que les fichiers Python déclarés existent et compilent."""
    files = [data["fichiers"][key] for key in PY_KEYS]
    for name in files:
        path = ROOT / name
        if not path.is_file():
            fail(f"fichier déclaré introuvable : {name}")
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            fail(f"erreur de syntaxe dans {name} : {exc.msg}")
    print(f"✓ fichiers Python présents et valides : {', '.join(files)}")
    return files


def check_screenshots():
    """Vérifie la présence des deux captures d'écran."""
    missing = [name for name in SCREENSHOTS if not (RENDU / name).is_file()]
    if missing:
        fail(f"capture(s) manquante(s) dans rendu/ : {', '.join(missing)}")
    print(f"✓ captures présentes : {', '.join(SCREENSHOTS)}")


def build_zip(data, py_files):
    """Construit l'archive avec exactement les fichiers demandés."""
    student_id = data["etudiant"]["numero_etudiant"].strip()
    archive = RENDU / f"{student_id}.zip"

    members = [(RENDU / "etape.json", "etape.json")]
    members += [(ROOT / name, name) for name in py_files]
    members += [(RENDU / name, name) for name in SCREENSHOTS]

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for source, name in members:
            zf.write(source, f"{student_id}/{name}")

    # Relecture de l'archive : rien d'autre que les fichiers attendus.
    with zipfile.ZipFile(archive) as zf:
        content = sorted(zf.namelist())
    expected = sorted(f"{student_id}/{name}" for _, name in members)
    if content != expected:
        fail(f"contenu inattendu dans l'archive : {content}")

    print(f"✓ archive créée : {archive.relative_to(ROOT)}")
    for name in content:
        print(f"    {name}")


def main():
    data = check_json()
    py_files = check_python_files(data)
    check_screenshots()
    build_zip(data, py_files)


if __name__ == "__main__":
    main()
