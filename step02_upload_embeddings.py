#!/usr/bin/env python3
"""
Step 02 - Upload des embeddings vers Hugging Face
Convertit les index FAISS en format SafeTensors et upload vers HF Hub
"""

import os
import json
import pickle
import getpass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse
import sys

def check_dependencies():
    """Vérifie les dépendances nécessaires."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
        
    try:
        from safetensors.torch import save_file, load_file
    except ImportError:
        missing.append("safetensors")
        
    try:
        from huggingface_hub import HfApi, create_repo, upload_file
    except ImportError:
        missing.append("huggingface-hub")
        
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    if missing:
        print(f"❌ Dépendances manquantes: {', '.join(missing)}")
        print("📦 Installer avec: pip install " + " ".join(missing))
        return False
    return True


class EmbeddingUploader:
    """Gestionnaire d'upload des embeddings vers Hugging Face Hub."""
    
    def __init__(self, hf_token: str):
        """
        Initialise l'uploader avec le token Hugging Face.
        
        Args:
            hf_token: Token d'accès Hugging Face
        """
        from huggingface_hub import HfApi
        
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)
        
    def convert_faiss_to_safetensors(self, faiss_index_path: str) -> Tuple[Dict, Dict]:
        """
        Convertit un index FAISS en format SafeTensors.
        
        Args:
            faiss_index_path: Chemin vers le répertoire d'index FAISS
            
        Returns:
            Tuple[Dict, Dict]: (tensors_dict, metadata_dict)
        """
        # Imports locaux pour éviter les erreurs si non installé
        import torch
        import numpy as np
        import faiss
        from safetensors.torch import save_file
        
        print(f"🔄 Conversion FAISS → SafeTensors depuis {faiss_index_path}")
        
        index_dir = Path(faiss_index_path)
        
        # Charger l'index FAISS
        index_file = index_dir / "index.faiss"
        metadata_file = index_dir / "metadata.json"
        mappings_file = index_dir / "mappings.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index FAISS non trouvé: {index_file}")
        
        print("  📂 Chargement index FAISS...")
        faiss_index = faiss.read_index(str(index_file))
        
        # Extraire les vecteurs depuis FAISS
        print("  📊 Extraction des vecteurs...")
        n_vectors = faiss_index.ntotal
        dimension = faiss_index.d
        
        print(f"     Vecteurs: {n_vectors:,}")
        print(f"     Dimension: {dimension}")
        
        # Reconstruction des vecteurs depuis l'index FAISS
        vectors = np.zeros((n_vectors, dimension), dtype=np.float32)
        
        if n_vectors > 0:
            # Pour les index HNSW, on doit reconstruire les vecteurs
            if hasattr(faiss_index, 'storage'):
                # Index avec storage (comme HNSW)
                storage = faiss_index.storage
                for i in range(n_vectors):
                    vectors[i] = storage.reconstruct(i)
            else:
                # Index flat
                vectors = faiss_index.reconstruct_n(0, n_vectors)
        
        # Charger les métadonnées JSON
        metadata_dict = {}
        if metadata_file.exists():
            print("  📋 Chargement métadonnées...")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
        
        # Charger les mappings
        id_to_idx = {}
        idx_to_id = {}
        if mappings_file.exists():
            print("  🔗 Chargement mappings...")
            with open(mappings_file, 'rb') as f:
                mappings = pickle.load(f)
                id_to_idx = mappings.get('id_to_idx', {})
                idx_to_id = mappings.get('idx_to_id', {})
        
        # Créer les tensors pour SafeTensors
        tensors_dict = {
            'embeddings': torch.from_numpy(vectors),
        }
        
        # Convertir les mappings en format compatible SafeTensors
        # Créer des tenseurs pour les IDs (en utilisant les hash des strings)
        if id_to_idx:
            print("  🔧 Conversion mappings...")
            
            # Créer une liste ordonnée des IDs par index
            ordered_ids = [''] * n_vectors
            for id_str, idx in id_to_idx.items():
                if idx < n_vectors:
                    ordered_ids[idx] = id_str
            
            # Encoder les IDs comme strings dans les métadonnées
            metadata_dict['ordered_ids'] = ordered_ids
            metadata_dict['id_to_idx'] = id_to_idx
        
        # Ajouter des métadonnées techniques
        metadata_dict.update({
            'format_version': '1.0',
            'total_vectors': n_vectors,
            'vector_dimension': dimension,
            'faiss_index_type': 'HNSW',
            'conversion_timestamp': datetime.now().isoformat(),
            'embedding_model': 'Qwen/Qwen3-Embedding-4B'  # À adapter selon config
        })
        
        print("✅ Conversion terminée")
        print(f"  📊 Tenseurs: {list(tensors_dict.keys())}")
        print(f"  📋 Métadonnées: {len(metadata_dict)} entrées")
        
        return tensors_dict, metadata_dict
    
    def upload_to_huggingface(self, 
                             tensors_dict: Dict,
                             metadata_dict: Dict,
                             repo_name: str,
                             dataset_name: str = "embeddings",
                             private: bool = True) -> str:
        """
        Upload les embeddings vers Hugging Face Hub.
        
        Args:
            tensors_dict: Dictionnaire des tenseurs
            metadata_dict: Métadonnées enrichies
            repo_name: Nom du repository (format: username/repo-name)
            dataset_name: Nom du dataset (défaut: embeddings)
            private: Repository privé ou public
            
        Returns:
            URL du repository créé
        """
        # Imports locaux pour HF Hub
        from safetensors.torch import save_file
        from huggingface_hub import create_repo, upload_file
        
        print(f"🚀 Upload vers Hugging Face Hub: {repo_name}")
        
        try:
            # Créer le repository s'il n'existe pas
            print("  📦 Création/vérification repository...")
            repo_url = create_repo(
                repo_id=repo_name,
                token=self.hf_token,
                private=private,
                exist_ok=True,
                repo_type="dataset"
            )
            print(f"  ✅ Repository: {repo_url}")
            
            # Créer des fichiers temporaires pour l'upload
            temp_dir = Path("./temp_upload")
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Sauvegarder les tenseurs en SafeTensors
                print("  💾 Sauvegarde SafeTensors...")
                safetensors_file = temp_dir / f"{dataset_name}.safetensors"
                save_file(tensors_dict, str(safetensors_file))
                
                # Sauvegarder les métadonnées en JSON
                print("  💾 Sauvegarde métadonnées...")
                metadata_file = temp_dir / f"{dataset_name}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
                
                # Créer un README pour le dataset
                print("  📝 Génération README...")
                readme_file = temp_dir / "README.md"
                self._create_dataset_readme(readme_file, repo_name, dataset_name, metadata_dict)
                
                # Upload des fichiers
                print("  📤 Upload fichiers...")
                
                files_to_upload = [
                    (safetensors_file, f"{dataset_name}.safetensors"),
                    (metadata_file, f"{dataset_name}_metadata.json"),
                    (readme_file, "README.md")
                ]
                
                for local_file, remote_path in files_to_upload:
                    print(f"    📤 {remote_path}...")
                    upload_file(
                        path_or_fileobj=str(local_file),
                        path_in_repo=remote_path,
                        repo_id=repo_name,
                        repo_type="dataset",
                        token=self.hf_token
                    )
                
                print("✅ Upload terminé avec succès !")
                
                # Informations de téléchargement
                print(f"\n📋 Informations du dataset:")
                print(f"  🔗 URL: https://huggingface.co/datasets/{repo_name}")
                print(f"  📊 Vecteurs: {metadata_dict.get('total_vectors', 'N/A'):,}")
                print(f"  📏 Dimension: {metadata_dict.get('vector_dimension', 'N/A')}")
                print(f"  💾 Taille SafeTensors: {safetensors_file.stat().st_size / 1024 / 1024:.1f} MB")
                
                # Sauvegarder la configuration pour Step 03
                self._save_step03_config(repo_name, dataset_name, metadata_dict)
                
                return repo_url
                
            finally:
                # Nettoyage des fichiers temporaires
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"❌ Erreur lors de l'upload: {e}")
            raise
    
    def _create_dataset_readme(self, readme_file: Path, repo_name: str, dataset_name: str, metadata_dict: Dict):
        """Créer un README descriptif pour le dataset."""
        content = f"""---
title: {repo_name}
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: static
app_file: README.md
pinned: false
license: apache-2.0
tags:
- embeddings
- faiss
- rag
- semantic-search
- safetensors
---

# 🔍 {repo_name} - Embeddings Dataset

## Description

Ce dataset contient des embeddings vectoriels générés par le système LocalRAG pour la recherche sémantique dans la documentation technique.

## 📊 Statistiques

- **Format**: SafeTensors
- **Vecteurs**: {metadata_dict.get('total_vectors', 'N/A'):,}
- **Dimension**: {metadata_dict.get('vector_dimension', 'N/A')}
- **Modèle d'embedding**: {metadata_dict.get('embedding_model', 'N/A')}
- **Type d'index**: {metadata_dict.get('faiss_index_type', 'N/A')}
- **Généré le**: {metadata_dict.get('conversion_timestamp', 'N/A')}

## 📁 Contenu

- `{dataset_name}.safetensors`: Embeddings vectoriels au format SafeTensors
- `{dataset_name}_metadata.json`: Métadonnées complètes avec mappings
- `README.md`: Cette documentation

## 🚀 Utilisation

### Chargement avec Hugging Face Hub

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

# Télécharger les fichiers
embeddings_file = hf_hub_download(repo_id="{repo_name}", filename="{dataset_name}.safetensors")
metadata_file = hf_hub_download(repo_id="{repo_name}", filename="{dataset_name}_metadata.json")

# Charger les embeddings
tensors = load_file(embeddings_file)
embeddings = tensors['embeddings']  # Shape: [n_vectors, dimension]

# Charger les métadonnées
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

print(f"Embeddings shape: {{embeddings.shape}}")
print(f"Total vectors: {{metadata['total_vectors']}}")
```

### Recherche sémantique

```python
import torch
import torch.nn.functional as F

def semantic_search(query_embedding, embeddings, top_k=10):
    \"\"\"Recherche sémantique dans les embeddings.\"\"\"
    # Calcul de similarité cosinus
    similarities = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings, dim=1)
    
    # Top-K résultats
    top_scores, top_indices = torch.topk(similarities, top_k)
    
    return top_indices, top_scores

# Exemple d'utilisation
query_emb = torch.randn(1, {metadata_dict.get('vector_dimension', 'dimension')})  # Votre embedding de requête
indices, scores = semantic_search(query_emb, embeddings)
```

## 🔧 Généré par

Ce dataset a été généré par [LocalRAG](https://github.com/your-repo/LocalRAG), un système RAG local complet pour la documentation technique.

- **Step 01**: Indexation vectorielle avec FAISS
- **Step 02**: Conversion SafeTensors et upload HF Hub
- **Step 03**: Recherche sémantique (à venir)
- **Step 04**: Génération RAG (à venir)

## 📝 License

Apache 2.0 - Voir LICENSE pour plus de détails.
"""
        
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _save_step03_config(self, repo_name: str, dataset_name: str, metadata_dict: Dict):
        """Sauvegarde la configuration pour Step 03."""
        config_file = Path("step03_config.json")
        
        # Configuration complète pour Step 03
        step03_config = {
            "step02_completed": True,
            "completion_timestamp": datetime.now().isoformat(),
            "huggingface": {
                "repo_id": repo_name,
                "dataset_name": dataset_name,
                "repo_type": "dataset",
                "files": {
                    "embeddings": f"{dataset_name}.safetensors",
                    "metadata": f"{dataset_name}_metadata.json",
                    "readme": "README.md"
                }
            },
            "embeddings_info": {
                "total_vectors": metadata_dict.get("total_vectors", 0),
                "vector_dimension": metadata_dict.get("vector_dimension", 0),
                "embedding_model": metadata_dict.get("embedding_model", "unknown"),
                "faiss_index_type": metadata_dict.get("faiss_index_type", "HNSW"),
                "format_version": metadata_dict.get("format_version", "1.0")
            },
            "usage_examples": {
                "download_command": f'hf_hub_download(repo_id="{repo_name}", filename="{dataset_name}.safetensors")',
                "load_command": f'tensors = load_file("{dataset_name}.safetensors"); embeddings = tensors["embeddings"]'
            }
        }
        
        # Sauvegarder avec formatage lisible
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(step03_config, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Configuration Step 03 sauvegardée: {config_file}")
            print(f"  📦 Repository HF: {repo_name}")
            print(f"  📊 Embeddings: {metadata_dict.get('total_vectors', 0):,} vecteurs")
            print(f"  📏 Dimension: {metadata_dict.get('vector_dimension', 0)}")
            print(f"  🔧 Prêt pour Step 03 !")
            
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde config Step 03: {e}")
            print(f"   Repository: {repo_name} (à noter manuellement)")
    
    def _create_dataset_readme(self, readme_file: Path, repo_name: str, dataset_name: str, metadata_dict: Dict):
        """Créer un README descriptif pour le dataset."""
        content = f"""---
title: {repo_name}
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: static
app_file: README.md
pinned: false
license: apache-2.0
tags:
- embeddings
- faiss
- rag
- semantic-search
- safetensors
---

# 🔍 {repo_name} - Embeddings Dataset

## Description

Ce dataset contient des embeddings vectoriels générés par le système LocalRAG pour la recherche sémantique dans la documentation technique.

## 📊 Statistiques

- **Format**: SafeTensors
- **Vecteurs**: {metadata_dict.get('total_vectors', 'N/A'):,}
- **Dimension**: {metadata_dict.get('vector_dimension', 'N/A')}
- **Modèle d'embedding**: {metadata_dict.get('embedding_model', 'N/A')}
- **Type d'index**: {metadata_dict.get('faiss_index_type', 'N/A')}
- **Généré le**: {metadata_dict.get('conversion_timestamp', 'N/A')}

## 📁 Contenu

- `{dataset_name}.safetensors`: Embeddings vectoriels au format SafeTensors
- `{dataset_name}_metadata.json`: Métadonnées complètes avec mappings
- `README.md`: Cette documentation

## 🚀 Utilisation

### Chargement avec Hugging Face Hub

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

# Télécharger les fichiers
embeddings_file = hf_hub_download(repo_id="{repo_name}", filename="{dataset_name}.safetensors")
metadata_file = hf_hub_download(repo_id="{repo_name}", filename="{dataset_name}_metadata.json")

# Charger les embeddings
tensors = load_file(embeddings_file)
embeddings = tensors['embeddings']  # Shape: [n_vectors, dimension]

# Charger les métadonnées
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

print(f"Embeddings shape: {{embeddings.shape}}")
print(f"Total vectors: {{metadata['total_vectors']}}")
```

## 📝 License

Apache 2.0 - Voir LICENSE pour plus de détails.
"""
        
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(content)


def get_user_inputs() -> Tuple[str, str, str, bool]:
    """Collecte les inputs utilisateur de manière interactive."""
    print("🔑 Configuration Hugging Face Hub")
    print("=" * 40)
    
    # Token HF
    token = getpass.getpass("🔐 Token Hugging Face (masqué): ")
    if not token.strip():
        raise ValueError("Token Hugging Face requis")
    
    # Nom du repository
    print("\n📦 Configuration du repository")
    repo_name = input("📛 Nom du repository (format: username/repo-name): ").strip()
    if not repo_name or '/' not in repo_name:
        raise ValueError("Format repository invalide (doit contenir username/repo-name)")
    
    # Nom du dataset
    dataset_name = input("📊 Nom du dataset [embeddings]: ").strip() or "embeddings"
    
    # Repository privé
    private_input = input("🔒 Repository privé ? (y/N): ").strip().lower()
    private = private_input in ['y', 'yes', 'oui']
    
    return token, repo_name, dataset_name, private


def find_faiss_indexes() -> List[Path]:
    """Trouve tous les répertoires d'index FAISS disponibles."""
    current_dir = Path(".")
    faiss_dirs = []
    
    # Chercher tous les dossiers contenant un index.faiss
    for item in current_dir.iterdir():
        if item.is_dir() and "faiss" in item.name.lower():
            index_file = item / "index.faiss"
            if index_file.exists():
                faiss_dirs.append(item)
    
    return sorted(faiss_dirs)

def select_faiss_index(provided_path: str = None) -> Path:
    """Sélectionne l'index FAISS à utiliser."""
    if provided_path:
        faiss_path = Path(provided_path)
        if faiss_path.exists() and (faiss_path / "index.faiss").exists():
            return faiss_path
        else:
            print(f"⚠️ Chemin fourni invalide: {faiss_path}")
    
    # Auto-détection des index disponibles
    available_indexes = find_faiss_indexes()
    
    if not available_indexes:
        print("❌ Aucun index FAISS trouvé dans le répertoire courant")
        print("💡 Lancez d'abord: python step01_indexer.py /path/to/docs")
        sys.exit(1)
    
    if len(available_indexes) == 1:
        selected = available_indexes[0]
        print(f"🎯 Index FAISS auto-détecté: {selected}")
        return selected
    
    # Plusieurs index disponibles, demander à l'utilisateur
    print(f"📂 {len(available_indexes)} index FAISS trouvés:")
    for i, idx_path in enumerate(available_indexes):
        # Lire quelques stats de l'index si possible
        try:
            metadata_file = idx_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    count = len(metadata)
                    print(f"  {i+1}. {idx_path} ({count:,} documents)")
            else:
                print(f"  {i+1}. {idx_path}")
        except:
            print(f"  {i+1}. {idx_path}")
    
    while True:
        try:
            choice = input(f"\nChoisissez un index (1-{len(available_indexes)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(available_indexes):
                selected = available_indexes[idx]
                print(f"✅ Index sélectionné: {selected}")
                return selected
            else:
                print(f"❌ Choix invalide. Entrez un nombre entre 1 et {len(available_indexes)}")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
        except KeyboardInterrupt:
            print("\n⏹️ Annulé par l'utilisateur")
            sys.exit(1)

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Upload des embeddings FAISS vers Hugging Face Hub")
    parser.add_argument("faiss_index_path", nargs="?", 
                       help="Chemin vers l'index FAISS (auto-détection si non spécifié)")
    parser.add_argument("--token", help="Token Hugging Face (ou utiliser input interactif)")
    parser.add_argument("--repo-name", help="Nom du repository (format: username/repo-name)")
    parser.add_argument("--dataset-name", default="embeddings", help="Nom du dataset")
    parser.add_argument("--private", action="store_true", help="Repository privé")
    parser.add_argument("--dry-run", action="store_true", help="Test de conversion sans upload")
    
    args = parser.parse_args()
    
    print("🚀 LocalRAG Step 02 - Upload des embeddings")
    print("=" * 50)
    
    # Vérification des dépendances seulement si pas --help
    if '--help' not in sys.argv and '-h' not in sys.argv:
        if not check_dependencies():
            sys.exit(1)
    
    # Sélection de l'index FAISS
    faiss_path = select_faiss_index(args.faiss_index_path)
    print(f"✅ Index FAISS confirmé: {faiss_path}")
    
    try:
        # Conversion FAISS → SafeTensors
        print(f"\n📋 Étape 1/3: Conversion")
        uploader = EmbeddingUploader("dummy_token")  # Token temporaire pour conversion
        tensors_dict, metadata_dict = uploader.convert_faiss_to_safetensors(str(faiss_path))
        
        if args.dry_run:
            print("\n✅ Test de conversion réussi (mode --dry-run)")
            print(f"📊 Vecteurs convertis: {metadata_dict['total_vectors']:,}")
            print(f"📏 Dimension: {metadata_dict['vector_dimension']}")
            return
        
        # Collecte des inputs utilisateur
        if args.token and args.repo_name:
            token, repo_name, dataset_name, private = args.token, args.repo_name, args.dataset_name, args.private
        else:
            print(f"\n📋 Étape 2/3: Configuration")
            token, repo_name, dataset_name, private = get_user_inputs()
        
        # Upload vers HF Hub
        print(f"\n📋 Étape 3/3: Upload")
        uploader = EmbeddingUploader(token)
        repo_url = uploader.upload_to_huggingface(
            tensors_dict, metadata_dict, repo_name, dataset_name, private
        )
        
        print("\n🎉 Upload terminé avec succès !")
        print(f"🔗 Repository: https://huggingface.co/datasets/{repo_name}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Opération annulée par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()