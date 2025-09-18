#!/usr/bin/env python3
"""
Utilitaires pour Step 03 - Lecture de la configuration Step 02
"""

import json
from pathlib import Path
from typing import Dict, Optional

class Step03Config:
    """Gestionnaire de configuration Step 03 basé sur la sortie Step 02."""
    
    def __init__(self, config_file: str = "step04_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Charge la configuration Step 03."""
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"❌ Configuration Step 03 non trouvée: {self.config_file}\n"
                f"💡 Lancez d'abord: python step02_upload_embeddings.py"
            )
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Vérification de la structure
            if not config.get("step02_completed"):
                raise ValueError("❌ Step 02 non complété selon la configuration")
            
            required_keys = ["huggingface", "embeddings_info"]
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"❌ Clé manquante dans configuration: {key}")
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Configuration Step 03 malformée: {e}")
    
    @property
    def repo_id(self) -> str:
        """Repository Hugging Face ID."""
        return self.config["huggingface"]["repo_id"]
    
    @property
    def dataset_name(self) -> str:
        """Nom du dataset."""
        return self.config["huggingface"]["dataset_name"]
    
    @property
    def embeddings_file(self) -> str:
        """Nom du fichier SafeTensors."""
        return self.config["huggingface"]["files"]["embeddings"]
    
    @property
    def metadata_file(self) -> str:
        """Nom du fichier métadonnées."""
        return self.config["huggingface"]["files"]["metadata"]
    
    @property
    def total_vectors(self) -> int:
        """Nombre total de vecteurs."""
        return self.config["embeddings_info"]["total_vectors"]
    
    @property
    def vector_dimension(self) -> int:
        """Dimension des vecteurs."""
        return self.config["embeddings_info"]["vector_dimension"]
    
    @property
    def embedding_model(self) -> str:
        """Modèle d'embedding utilisé."""
        return self.config["embeddings_info"]["embedding_model"]
    
    @property
    def download_command(self) -> str:
        """Commande de téléchargement HF Hub."""
        return self.config["usage_examples"]["download_command"]
    
    @property
    def load_command(self) -> str:
        """Commande de chargement SafeTensors."""
        return self.config["usage_examples"]["load_command"]
    
    def print_summary(self):
        """Affiche un résumé de la configuration."""
        print("📋 Configuration Step 03 - Résumé")
        print("=" * 40)
        print(f"📦 Repository HF: {self.repo_id}")
        print(f"📊 Embeddings: {self.total_vectors:,} vecteurs")
        print(f"📏 Dimension: {self.vector_dimension}")
        print(f"🧠 Modèle: {self.embedding_model}")
        print(f"📁 Fichier: {self.embeddings_file}")
        print(f"⏰ Complété: {self.config.get('completion_timestamp', 'N/A')}")
        print()
        print("🚀 Prêt pour la recherche sémantique !")
    
    def get_download_instructions(self) -> Dict[str, str]:
        """Retourne les instructions de téléchargement."""
        return {
            "python_code": f'''
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

# Télécharger les fichiers
embeddings_file = hf_hub_download(
    repo_id="{self.repo_id}", 
    filename="{self.embeddings_file}"
)
metadata_file = hf_hub_download(
    repo_id="{self.repo_id}", 
    filename="{self.metadata_file}"
)

# Charger les embeddings
tensors = load_file(embeddings_file)
embeddings = tensors["embeddings"]  # Shape: [{self.total_vectors}, {self.vector_dimension}]

# Charger les métadonnées
with open(metadata_file, 'r') as f:
    metadata = json.load(f)
    
print(f"✅ Embeddings chargés: {{embeddings.shape}}")
'''.strip(),
            "cli_download": f"huggingface-cli download {self.repo_id} --repo-type dataset",
            "repo_url": f"https://huggingface.co/datasets/{self.repo_id}"
        }


def load_step04_config(config_file: str = "step04_config.json") -> Step03Config:
    """
    Fonction utilitaire pour charger la configuration Step 03.
    
    Args:
        config_file: Chemin vers le fichier de configuration
        
    Returns:
        Instance de Step03Config
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si la configuration est invalide
    """
    return Step03Config(config_file)


def check_step04_ready() -> bool:
    """
    Vérifie si Step 03 peut être lancé (configuration Step 02 disponible).
    
    Returns:
        True si prêt, False sinon
    """
    try:
        config = load_step04_config()
        return config.config.get("step02_completed", False)
    except (FileNotFoundError, ValueError):
        return False


if __name__ == "__main__":
    """Test de la configuration Step 03."""
    try:
        print("🧪 Test de configuration Step 03")
        print("=" * 40)
        
        if check_step04_ready():
            config = load_step04_config()
            config.print_summary()
            
            print("\n📖 Instructions de téléchargement:")
            instructions = config.get_download_instructions()
            print(instructions["python_code"])
            
        else:
            print("❌ Step 03 non prêt")
            print("💡 Lancez d'abord: python step02_upload_embeddings.py")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")