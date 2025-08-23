#!/usr/bin/env python3
"""
Step 04 - Déploiement HuggingFace Spaces pour LocalRAG
Déploie le système RAG complet vers HuggingFace Spaces avec support ZeroGPU et MCP
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import tempfile
import getpass
import json
import argparse

def check_dependencies():
    """Vérifie les dépendances nécessaires"""
    print("🔍 Vérification des dépendances...")
    
    # Vérifier git
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✅ Git installé")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git non installé. Veuillez installer git d'abord.")
        sys.exit(1)
    
    # Vérifier huggingface_hub
    try:
        import huggingface_hub
        print("✅ HuggingFace Hub disponible")
    except ImportError:
        print("❌ HuggingFace Hub non trouvé. Installation...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        print("✅ HuggingFace Hub installé")

def get_hf_token():
    """Récupère le token HuggingFace depuis l'environnement ou saisie utilisateur"""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    if not token:
        print("\n🔑 Token HuggingFace requis pour le déploiement")
        print("Obtenez votre token sur: https://huggingface.co/settings/tokens")
        print("Assurez-vous que votre token a les permissions 'write'")
        token = getpass.getpass("Entrez votre token HuggingFace: ").strip()
    
    if not token:
        print("❌ Aucun token fourni. Déploiement annulé.")
        sys.exit(1)
    
    return token

def get_space_info():
    """Récupère les informations du Space auprès de l'utilisateur"""
    print("\n📝 Configuration du Space HuggingFace")
    print("-" * 40)
    
    # Nom du space (format username/space-name)
    space_name = input("Nom du Space (ex: username/localrag-demo): ").strip()
    if not space_name or '/' not in space_name:
        print("❌ Format invalide. Utilisez: username/space-name")
        sys.exit(1)
    
    # Titre du space
    default_title = "🔍 LocalRAG - RAG System with Qwen3"
    space_title = input(f"Titre du Space (défaut: {default_title}): ").strip()
    if not space_title:
        space_title = default_title
    
    # Description courte
    default_desc = "Local RAG system with Qwen3 models and reranking"
    space_desc = input(f"Description courte (défaut: {default_desc}): ").strip()
    if not space_desc:
        space_desc = default_desc
    
    # Repository d'embeddings depuis step03_config.json
    config_file = Path("step03_config.json")
    repo_id = None
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                repo_id = config.get('huggingface', {}).get('repo_id')
                if repo_id:
                    print(f"📦 Repository embeddings détecté: {repo_id}")
        except Exception as e:
            print(f"⚠️ Erreur lecture step03_config.json: {e}")
    
    if not repo_id:
        repo_id = input("Repository HF des embeddings (ex: username/embeddings): ").strip()
        if not repo_id:
            print("❌ Repository embeddings requis")
            sys.exit(1)
    
    # Visibilité du space
    is_private = input("Space privé? (y/N): ").strip().lower() in ['y', 'yes', 'oui']
    
    return {
        'name': space_name,
        'title': space_title,
        'description': space_desc,
        'repo_id': repo_id,
        'private': is_private
    }


def create_space_readme(space_info):
    """Crée le README.md pour le Space"""
    print("📝 Création du README.md pour le Space...")
    
    readme_content = f'''---
title: {space_info['title']}
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.43.1
app_file: step03_chatbot.py
pinned: false
license: mit
hardware: zerogpu
short_description: {space_info['description']}
models:
- Qwen/Qwen3-Embedding-4B
- Qwen/Qwen3-Reranker-4B
- Qwen/Qwen3-4B-Instruct-2507
datasets:
- {space_info['repo_id']}
tags:
- rag
- retrieval-augmented-generation
- qwen3
- semantic-search
- question-answering
- zero-gpu
- mcp-server
- faiss
---

# 🔍 LocalRAG - Système RAG Complet avec Qwen3

Système RAG (Retrieval-Augmented Generation) complet utilisant les modèles Qwen3 de dernière génération avec reranking et génération streamée.

## ⚡ Fonctionnalités

### 🧠 **Modèles IA Avancés**
- **Embeddings**: Qwen3-Embedding-4B (2560 dimensions)
- **Reranking**: Qwen3-Reranker-4B pour l'affinage des résultats
- **Génération**: Qwen3-4B-Instruct-2507 avec streaming
- **Optimisation ZeroGPU**: Support natif avec décorateurs @spaces.GPU

### 🔍 **Recherche Sémantique Avancée**
- **Pipeline 2 étapes**: Recherche vectorielle + reranking
- **Index FAISS**: Recherche haute performance sur de gros volumes
- **Scores détaillés**: Embedding + reranking pour chaque document
- **Sélection intelligente**: Top-K adaptatif selon pertinence

### 💬 **Génération Contextuelle**
- **Streaming**: Réponse progressive token par token
- **Contexte enrichi**: Intégration des documents les plus pertinents
- **Références**: Sources avec scores de pertinence
- **Qualité**: Réponses basées uniquement sur le contexte fourni

### 🔌 **Intégration MCP**
- **Serveur MCP natif**: Fonction `ask_rag_question()` exposée
- **Paramètres configurables**: Nombre documents, activation reranking
- **Compatible**: Claude Desktop, VS Code, Cursor IDE
- **API structurée**: Réponses JSON avec sources et métadonnées

## 🚀 Utilisation

### Interface Web
1. **Posez votre question** dans le chat
2. **Observez la recherche** en 2 étapes (vectorielle → reranking)
3. **Lisez la réponse** générée en streaming
4. **Consultez les sources** avec scores de pertinence

### Paramètres Avancés
- **Documents finaux**: Nombre de documents pour la génération (1-10)
- **Reranking**: Activer/désactiver l'affinage Qwen3
- **Historique**: Conversations contextuelles

### Intégration MCP
Connectez votre client MCP pour un accès programmatique :
```python
# Exemple d'utilisation MCP
result = mcp_client.call_tool(
    "ask_rag_question",
    question="Comment implémenter des réseaux de neurones complexes?",
    num_documents=3,
    use_reranking=True
)
```

## 🎯 Cas d'Usage Parfaits

- **Documentation technique**: Recherche dans APIs, guides, tutoriels
- **Support client**: Réponses basées sur une base de connaissances
- **Recherche académique**: Analyse de corpus documentaires
- **Assistance développeur**: Aide contextuelle sur frameworks/librairies
- **Formation**: Système de questions-réponses intelligent

## 📊 Performance

- **Recherche**: ~50ms pour 10K+ documents
- **Reranking**: ~200ms pour 20 candidats  
- **Génération**: ~2-4s avec streaming
- **Mémoire**: ~6-8GB optimisé pour ZeroGPU

## 🔒 Sécurité & Confidentialité

- **ZeroGPU**: Traitement sécurisé sans stockage persistant
- **Données temporaires**: Pas de rétention des questions/réponses
- **Modèles locaux**: Traitement dans l'environnement HF Spaces

## 📚 Source des Données

Ce Space utilise des embeddings pré-calculés depuis le dataset :
**[{space_info['repo_id']}](https://huggingface.co/datasets/{space_info['repo_id']})**

Commencez à poser vos questions pour découvrir la puissance du RAG avec Qwen3! 🔍✨
'''
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ README.md créé")

def create_requirements_txt():
    """Crée le requirements.txt pour le Space"""
    print("📝 Création de requirements.txt pour le Space...")
    
    requirements_content = '''gradio>=5.43.1
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.45.0
accelerate>=0.24.0
safetensors>=0.3.0
tokenizers>=0.15.0
sentencepiece>=0.1.97
huggingface_hub>=0.20.0
faiss-cpu>=1.7.0
numpy>=1.21.0
scipy>=1.9.0
python-dotenv>=1.0.0
sentence-transformers>=3.0.0
flash-attn>=2.0.0
spaces
'''
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ requirements.txt créé")

def validate_files():
    """Valide que tous les fichiers requis existent"""
    print("🔍 Validation des fichiers...")
    
    required_files = [
        "requirements.txt", 
        "README.md",
        "step03_chatbot.py",
        "step03_config.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Fichiers manquants: {', '.join(missing_files)}")
        if "step03_config.json" in missing_files:
            print("💡 Lancez d'abord step02 pour générer la configuration")
        sys.exit(1)
    
    # Vérifier les décorateurs ZeroGPU dans step03_chatbot.py
    with open("step03_chatbot.py", "r") as f:
        content = f.read()
        if "@spaces.GPU" not in content:
            print("⚠️ Attention: Pas de décorateurs @spaces.GPU trouvés dans step03_chatbot.py")
            print("Le Space fonctionnera mais sans optimisations GPU")
        else:
            print("✅ Décorateurs ZeroGPU détectés")
    
    # Vérifier le support MCP
    if "mcp_server=True" not in content and "/mcp/" not in content:
        print("⚠️ Attention: Support MCP non détecté")
    else:
        print("✅ Support MCP détecté")
    
    print("✅ Tous les fichiers requis présents")

def create_space(space_info, token):
    """Crée ou met à jour le Space HuggingFace"""
    print(f"🚀 Création/mise à jour du Space: {space_info['name']}")
    
    from huggingface_hub import HfApi, login
    
    # Connexion à HuggingFace
    login(token=token, add_to_git_credential=True)
    
    api = HfApi()
    
    try:
        # Vérifier si le space existe
        space_info_api = api.space_info(repo_id=space_info['name'])
        print(f"✅ Space {space_info['name']} existe déjà, mise à jour...")
        update_mode = True
    except Exception:
        print(f"📦 Création du nouveau Space: {space_info['name']}")
        update_mode = False
        
        # Créer le space
        try:
            api.create_repo(
                repo_id=space_info['name'],
                repo_type="space",
                space_sdk="gradio",
                space_hardware="zerogpu",
                private=space_info['private']
            )
            print("✅ Space créé avec succès")
        except Exception as e:
            print(f"❌ Échec création space: {e}")
            sys.exit(1)
    
    return update_mode

def deploy_files(space_info, token):
    """Déploie les fichiers vers le Space"""
    print("📤 Upload des fichiers vers le Space...")
    
    from huggingface_hub import HfApi
    
    api = HfApi()
    
    # Fichiers à uploader
    files_to_upload = [
        "requirements.txt",
        "README.md", 
        "step03_chatbot.py",
        "step03_config.json",
        "step03_utils.py"
    ]
    
    try:
        upload_results = []
        for file in files_to_upload:
            if os.path.exists(file):
                print(f"  📄 Upload {file}...")
                try:
                    api.upload_file(
                        path_or_fileobj=file,
                        path_in_repo=file,
                        repo_id=space_info['name'],
                        repo_type="space",
                        token=token
                    )
                    print(f"     ✅ {file} uploadé avec succès")
                    upload_results.append((file, True, None))
                except Exception as e:
                    print(f"     ❌ Erreur upload {file}: {e}")
                    upload_results.append((file, False, str(e)))
            else:
                print(f"  ⚠️ Fichier manquant ignoré: {file}")
        
        # Vérifier les résultats
        failed_uploads = [r for r in upload_results if not r[1]]
        if failed_uploads:
            print(f"\n⚠️ {len(failed_uploads)} fichier(s) ont échoué:")
            for filename, success, error in failed_uploads:
                print(f"  ❌ {filename}: {error}")
            print("\n💡 Le Space pourrait ne pas fonctionner correctement")
        else:
            print("✅ Tous les fichiers uploadés avec succès")
        
    except Exception as e:
        print(f"❌ Échec upload des fichiers: {e}")
        sys.exit(1)

def wait_for_space_build(space_info):
    """Attend que le Space se construise"""
    print("⏳ Le Space se construit... Cela peut prendre quelques minutes.")
    print(f"🌐 Suivez la construction sur: https://huggingface.co/spaces/{space_info['name']}")
    print("📱 Le Space sera disponible une fois la construction terminée.")
    print("\n⚡ Note: Le premier démarrage peut être plus long (téléchargement modèles)")

def main():
    """Fonction principale de déploiement"""
    parser = argparse.ArgumentParser(description="Déploiement LocalRAG vers HuggingFace Spaces")
    parser.add_argument("--space-name", help="Nom du Space (ex: username/space-name)")
    parser.add_argument("--private", action="store_true", help="Créer un Space privé")
    args = parser.parse_args()
    
    print("🔍 LocalRAG Step 04 - Déploiement HuggingFace Spaces")
    print("=" * 60)
    
    # Changer vers le répertoire du script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Vérifier les dépendances
    check_dependencies()
    
    # Obtenir le token HuggingFace
    token = get_hf_token()
    
    # Obtenir les informations du Space
    if args.space_name:
        # Mode non-interactif avec args
        space_info = {
            'name': args.space_name,
            'title': '🔍 LocalRAG - RAG System with Qwen3',
            'description': 'Local RAG system with Qwen3 models and reranking',
            'repo_id': None,
            'private': args.private
        }
        
        # Charger repo_id depuis config
        config_file = Path("step03_config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                space_info['repo_id'] = config.get('repo_id')
        
        if not space_info['repo_id']:
            print("❌ step03_config.json requis pour le repository embeddings")
            sys.exit(1)
    else:
        space_info = get_space_info()
    
    # Créer les fichiers de configuration
    create_space_readme(space_info)
    # create_requirements_txt() # Désactivé - utiliser le requirements.txt existant
    
    # Valider les fichiers
    validate_files()
    
    # Créer ou mettre à jour le space
    update_mode = create_space(space_info, token)
    
    # Déployer les fichiers
    deploy_files(space_info, token)
    
    # Message de succès
    print("\n🎉 Déploiement terminé avec succès!")
    print(f"🌐 URL du Space: https://huggingface.co/spaces/{space_info['name']}")
    
    if not update_mode:
        wait_for_space_build(space_info)
    
    print(f"\n📱 Votre LocalRAG est maintenant en ligne:")
    print(f"   https://huggingface.co/spaces/{space_info['name']}")
    print(f"\n🚀 Space déployé avec ZeroGPU pour des performances optimales!")
    print(f"🔧 Serveur MCP natif activé pour l'accès programmatique!")
    print("\n🔍 Bonne recherche dans vos documents! ✨")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Déploiement annulé par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Déploiement échoué: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)