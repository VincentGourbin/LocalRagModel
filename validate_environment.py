#!/usr/bin/env python3
"""
Validation de l'environnement LocalRAG
Vérifie que toutes les dépendances sont correctement installées
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Vérifie la version Python"""
    print("🐍 Vérification Python...")
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ requis (détecté: {version.major}.{version.minor})")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu_availability():
    """Vérifie la disponibilité GPU"""
    print("\n🖥️  Vérification GPU...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # MPS (Mac)
        if torch.backends.mps.is_available():
            print("✅ MPS (Mac GPU) disponible")
            return True
        
        # CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA disponible ({gpu_count} GPU)")
            print(f"   GPU 0: {gpu_name}")
            return True
            
        print("❌ Aucun GPU détecté (MPS/CUDA)")
        print("   Le script d'indexation nécessite un GPU")
        return False
        
    except ImportError:
        print("❌ PyTorch non installé")
        return False

def check_dependencies():
    """Vérifie les dépendances principales"""
    print("\n📦 Vérification des dépendances...")
    
    dependencies = [
        ("faiss-cpu", "faiss"),
        ("sentence-transformers", "sentence_transformers"), 
        ("transformers", "transformers"),
        ("beautifulsoup4", "bs4"),
        ("requests", "requests"),
        ("pillow", "PIL"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
        ("markdown", "markdown"),
        ("ollama", "ollama"),
        ("huggingface-hub", "huggingface_hub"),
        ("safetensors", "safetensors")
    ]
    
    all_ok = True
    for package_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Installer avec: pip install {package_name}")
            all_ok = False
            
    return all_ok

def check_ollama():
    """Vérifie Ollama pour l'analyse d'images"""
    print("\n🦙 Vérification Ollama...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama actif ({len(models)} modèles installés)")
            
            # Vérifier llava pour l'analyse d'images
            llava_found = any("llava" in model.get("name", "") for model in models)
            if llava_found:
                print("✅ Modèle llava trouvé (analyse d'images)")
            else:
                print("⚠️ Modèle llava non trouvé")
                print("   Installer avec: ollama pull llava")
            return True
        else:
            print("❌ Ollama non accessible")
            return False
            
    except Exception as e:
        print("❌ Ollama non accessible")
        print("   Démarrer avec: ollama serve")
        return False

def check_project_structure():
    """Vérifie la structure du projet"""
    print("\n📁 Vérification structure projet...")
    
    required_files = [
        "step01_indexer.py",
        "step02_upload_embeddings.py",
        "faiss_indexer.py", 
        "requirements.txt",
        "README.md"
    ]
    
    all_ok = True
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} manquant")
            all_ok = False
            
    return all_ok

def main():
    """Validation complète"""
    print("🔍 LocalRAG - Validation de l'environnement")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_project_structure(),
        check_dependencies(),
        check_gpu_availability(),
        check_ollama()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 Environnement validé avec succès !")
        print("\n📋 Étapes suivantes :")
        print("1. Copier config_example.py vers config.py")
        print("2. Personnaliser config.py selon vos besoins")
        print("3. Lancer l'indexation:")
        print("   python step01_indexer.py /path/to/docs")
    else:
        failed = sum(1 for check in checks if not check)
        print(f"❌ {failed} problème(s) détecté(s)")
        print("Corrigez les erreurs avant de continuer")
        sys.exit(1)

if __name__ == "__main__":
    main()