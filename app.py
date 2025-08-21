#!/usr/bin/env python3
"""
Point d'entrée pour ZeroGPU Hugging Face Space
Compatible avec l'exécution locale et ZeroGPU
"""

import sys
import os

# Configuration pour ZeroGPU
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"

def main():
    """Point d'entrée principal compatible ZeroGPU et local"""
    print("🚀 Démarrage LocalRAG Step 03")
    
    # Détection d'environnement
    if os.getenv("SPACE_ID"):
        print("🌐 Environnement: Hugging Face Space ZeroGPU")
        print(f"📦 Space ID: {os.getenv('SPACE_ID')}")
    else:
        print("💻 Environnement: Local")
    
    # Import et lancement du chatbot
    try:
        from step03_chatbot import main as chatbot_main
        return chatbot_main()
    except Exception as e:
        print(f"❌ Erreur de démarrage: {e}")
        return 1

if __name__ == "__main__":
    exit(main())