#!/usr/bin/env python3
"""
Setup script for LocalRAG
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description):
    """Execute a shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erreur: {e.stderr}")
        return False

def detect_system():
    """Detect system and GPU capabilities"""
    system = platform.system()
    print(f"🖥️  Système détecté: {system}")
    
    gpu_info = {}
    
    # Check for Mac M1/M2/M3
    if system == "Darwin":
        try:
            import torch
            if torch.backends.mps.is_available():
                gpu_info["type"] = "MPS"
                gpu_info["available"] = True
                print("✅ Mac avec puce Apple Silicon (MPS disponible)")
            else:
                gpu_info["type"] = "None"
                gpu_info["available"] = False
                print("⚠️ Mac sans support MPS")
        except ImportError:
            print("⚠️ PyTorch non installé - impossible de vérifier MPS")
            gpu_info["type"] = "Unknown"
            gpu_info["available"] = False
    
    # Check for NVIDIA CUDA
    elif system in ["Linux", "Windows"]:
        try:
            result = subprocess.run("nvidia-smi", capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info["type"] = "CUDA"
                gpu_info["available"] = True
                print("✅ GPU NVIDIA détecté (CUDA disponible)")
            else:
                gpu_info["type"] = "None"
                gpu_info["available"] = False
                print("⚠️ Aucun GPU NVIDIA détecté")
        except FileNotFoundError:
            gpu_info["type"] = "None"
            gpu_info["available"] = False
            print("⚠️ NVIDIA drivers non installés")
    
    return system, gpu_info

def install_python_packages(system_type, gpu_info):
    """Install Python packages with system-specific optimizations"""
    print("\n📦 Installation des packages Python...")
    
    # Base installation
    if not run_command("pip install --upgrade pip", "Mise à jour pip"):
        return False
    
    # Install PyTorch with appropriate backend
    if gpu_info["type"] == "CUDA":
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        if not run_command(torch_cmd, "Installation PyTorch (CUDA)"):
            return False
    elif gpu_info["type"] == "MPS":
        if not run_command("pip install torch torchvision torchaudio", "Installation PyTorch (MPS)"):
            return False
    else:
        print("⚠️ Installation PyTorch CPU (performance limitée)")
        if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", "Installation PyTorch (CPU)"):
            return False
    
    # Install other requirements
    if not run_command("pip install -r requirements.txt", "Installation dépendances"):
        return False
    
    return True

def setup_ollama():
    """Setup Ollama for image analysis"""
    print("\n🦙 Configuration Ollama...")
    
    # Check if ollama is installed
    try:
        result = subprocess.run("ollama --version", capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✅ Ollama déjà installé")
        else:
            print("❌ Ollama non trouvé")
            return False
    except FileNotFoundError:
        print("❌ Ollama non installé")
        print("📋 Installation manuelle requise:")
        print("   macOS: brew install ollama")
        print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   Windows: Télécharger depuis https://ollama.ai")
        return False
    
    # Start ollama service
    if not run_command("ollama serve &", "Démarrage service Ollama"):
        print("⚠️ Le service Ollama doit être démarré manuellement")
    
    # Pull required model
    if run_command("ollama pull llava", "Téléchargement modèle llava"):
        print("✅ Modèle llava installé pour l'analyse d'images")
    else:
        print("⚠️ Échec téléchargement llava - l'analyse d'images sera limitée")
    
    return True

def create_config():
    """Create default configuration"""
    print("\n⚙️ Création configuration par défaut...")
    
    config_path = Path("config.py")
    if config_path.exists():
        print("✅ config.py existe déjà")
        return True
    
    try:
        import shutil
        shutil.copy("config_example.py", "config.py")
        print("✅ config.py créé depuis config_example.py")
        print("📝 Éditez config.py pour personnaliser les paramètres")
        return True
    except Exception as e:
        print(f"❌ Erreur création config.py: {e}")
        return False

def main():
    """Main setup process"""
    print("🚀 LocalRAG - Installation automatique")
    print("=" * 50)
    
    # System detection
    system, gpu_info = detect_system()
    
    if not gpu_info["available"]:
        print("\n⚠️ ATTENTION: Aucun GPU détecté")
        print("Le système fonctionnera en mode CPU avec des performances limitées")
        response = input("Continuer quand même ? (y/N): ")
        if response.lower() != 'y':
            print("Installation annulée")
            sys.exit(1)
    
    # Installation steps
    steps = [
        ("Python packages", lambda: install_python_packages(system, gpu_info)),
        ("Ollama setup", setup_ollama),
        ("Configuration", create_config)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n📋 Étape: {step_name}")
        if step_func():
            success_count += 1
        else:
            print(f"❌ Échec étape: {step_name}")
    
    print("\n" + "=" * 50)
    
    if success_count == len(steps):
        print("🎉 Installation terminée avec succès !")
        print("\n📋 Prochaines étapes:")
        print("1. python validate_environment.py  # Valider l'installation")
        print("2. Éditer config.py selon vos besoins")
        print("3. python step01_indexer.py /path/to/docs  # Commencer l'indexation")
    else:
        print(f"⚠️ Installation partiellement réussie ({success_count}/{len(steps)} étapes)")
        print("Vérifiez les erreurs ci-dessus et relancez si nécessaire")

if __name__ == "__main__":
    main()