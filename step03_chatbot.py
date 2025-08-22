#!/usr/bin/env python3
"""
Step 03 - Interface de chat RAG générique avec Gradio
Utilise les embeddings de Step 02 depuis Hugging Face Hub + Qwen3-4B-Instruct-2507 pour génération
"""

import os
import json
import numpy as np
import gradio as gr
from gradio import ChatMessage
from typing import List, Dict, Optional, Tuple
import time
import torch
import threading
import http.server
import socketserver
from pathlib import Path
from datetime import datetime

# ZeroGPU compatibility
try:
    import spaces
    ZEROGPU_AVAILABLE = True
    print("🚀 ZeroGPU détecté - activation du support")
except ImportError:
    ZEROGPU_AVAILABLE = False
    # Fallback decorator for local usage
    class MockSpaces:
        @staticmethod
        def GPU(duration=None):
            def decorator(func):
                return func
            return decorator
    spaces = MockSpaces()

def _check_dependencies():
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
        from safetensors.torch import load_file
    except ImportError:
        missing.append("safetensors")
        
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        missing.append("huggingface-hub")
        
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
        
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    except ImportError:
        missing.append("transformers")
        
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")
    
    if missing:
        print(f"❌ Dépendances manquantes: {', '.join(missing)}")
        print("📦 Installer avec: pip install " + " ".join(missing))
        return False
    return True


class Step03Config:
    """Gestionnaire de configuration Step 03 basé sur la sortie Step 02."""
    
    def __init__(self, config_file: str = "step03_config.json"):
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


class Qwen3Reranker:
    """
    Reranker utilisant Qwen3-Reranker-4B pour améliorer la pertinence des résultats de recherche
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-4B", use_flash_attention: bool = True):
        """
        Initialise le reranker Qwen3
        
        Args:
            model_name: Nom du modèle HuggingFace à charger
            use_flash_attention: Utiliser Flash Attention 2 si disponible (auto-désactivé sur Mac)
        """
        self.model_name = model_name
        self.use_flash_attention = use_flash_attention
        
        # Détection de l'environnement
        self.is_mps = torch.backends.mps.is_available()
        self.is_cuda = torch.cuda.is_available()
        self.is_cpu = not self.is_mps and not self.is_cuda
        
        print(f"🔄 Chargement du reranker {model_name}...")
        self._detect_platform()
        self._load_model()
    
    def _detect_platform(self):
        """Détecte la plateforme et ajuste les paramètres"""
        if self.is_mps:
            print("  - Plateforme: Mac MPS détecté")
            self.use_flash_attention = False  # Flash Attention non compatible MPS
            self.batch_size = 1  # Traitement strictement individuel sur Mac
            self.memory_cleanup_freq = 3  # Nettoyage mémoire fréquent
        elif self.is_cuda:
            print(f"  - Plateforme: CUDA détecté ({torch.cuda.get_device_name()})")
            self.batch_size = 1  # Garde traitement individuel pour stabilité
            self.memory_cleanup_freq = 10  # Nettoyage moins fréquent
        else:
            print("  - Plateforme: CPU")
            self.use_flash_attention = False
            self.batch_size = 1
            self.memory_cleanup_freq = 5
    
    def _load_model(self):
        """Charge le modèle et le tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Chargement du tokenizer
            print("  - Chargement du tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configuration du modèle selon la plateforme
            model_kwargs = self._get_model_config()
            
            # Chargement du modèle
            print("  - Chargement du modèle...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Configuration du device
            self._setup_device()
            
            print(f"✅ Reranker chargé sur {self.device}")
            print(f"  - Flash Attention: {'✅' if self.use_flash_attention else '❌'}")
            print(f"  - Paramètres: {self.get_parameter_count():.1f}B")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du reranker: {e}")
            print("💡 Le reranking sera désactivé")
            self.model = None
            self.tokenizer = None
            self.device = None
    
    def _get_model_config(self) -> Dict:
        """Retourne la configuration du modèle selon la plateforme"""
        config = {}
        
        if self.is_mps:
            # Configuration pour Mac MPS
            config["torch_dtype"] = torch.float32  # MPS fonctionne mieux avec float32
            config["device_map"] = None  # device_map peut causer des problèmes avec MPS
        elif self.is_cuda:
            # Configuration pour CUDA
            config["torch_dtype"] = torch.float16
            if self.use_flash_attention:
                try:
                    config["attn_implementation"] = "flash_attention_2"
                    print("  - Flash Attention 2 activée")
                except Exception:
                    print("  - Flash Attention 2 non disponible, utilisation standard")
                    self.use_flash_attention = False
            else:
                config["device_map"] = "auto"
        else:
            # Configuration pour CPU
            config["torch_dtype"] = torch.float32
            config["device_map"] = "cpu"
        
        return config
    
    def _setup_device(self):
        """Configure le device pour le modèle"""
        if self.is_mps:
            self.device = torch.device("mps")
            self.model = self.model.to(self.device)
        elif self.is_cuda:
            if hasattr(self.model, 'device'):
                self.device = next(self.model.parameters()).device
            else:
                self.device = torch.device("cuda")
                self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
    
    def _format_pair(self, query: str, document: str, instruction: str = None) -> str:
        """
        Formate une paire query-document pour le reranker
        """
        if instruction:
            return f"Instruction: {instruction}\nQuery: {query}\nDocument: {document}"
        return f"Query: {query}\nDocument: {document}"
    
    def _get_default_instruction(self) -> str:
        """Retourne l'instruction par défaut pour la documentation technique"""
        return (
            "Évaluez la pertinence de ce document technique "
            "par rapport à la requête en considérant : terminologie technique, "
            "spécifications, normes, procédures de mise en œuvre."
        )
    
    def _process_single_document(self, query: str, document: str, instruction: str) -> float:
        """
        Traite un seul document et retourne son score de pertinence
        """
        # Formatage de la paire
        pair_text = self._format_pair(query, document, instruction)
        
        # Tokenisation (pas de problème de padding avec un seul document)
        inputs = self.tokenizer(
            pair_text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            padding=False
        )
        
        # Déplacement vers le device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inférence
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Le modèle Qwen3-Reranker retourne des logits de forme [1, 2]
            # pour classification binaire : [non-pertinent, pertinent]
            probs = torch.nn.functional.softmax(logits, dim=1)
            score = probs[0, 1].cpu().item()  # Classe 1 = pertinent
            
            return float(score)
    
    def _cleanup_memory(self):
        """Nettoie la mémoire selon la plateforme"""
        if self.is_mps:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        elif self.is_cuda:
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
    
    @spaces.GPU(duration=60)  # ZeroGPU: alloue GPU pour 60s max pour reranking
    def rerank(self, query: str, documents: List[str], instruction: str = None) -> List[float]:
        """
        Reranke une liste de documents par rapport à une requête
        """
        if not documents:
            return []
        
        if self.model is None or self.tokenizer is None:
            print("  - Reranker non disponible, scores neutres retournés")
            return [0.5] * len(documents)
        
        if instruction is None:
            instruction = self._get_default_instruction()
        
        print(f"  - Reranking de {len(documents)} documents (traitement individuel)")
        
        scores = []
        successful_count = 0
        
        for i, document in enumerate(documents):
            try:
                score = self._process_single_document(query, document, instruction)
                score = max(0.0, min(1.0, score))
                scores.append(score)
                successful_count += 1
                
                if (i + 1) % self.memory_cleanup_freq == 0:
                    self._cleanup_memory()
                
            except Exception as doc_error:
                print(f"    ⚠️ Erreur document {i+1}: {doc_error}")
                scores.append(0.5)  # Score neutre en cas d'erreur
        
        self._cleanup_memory()
        
        print(f"  ✅ Reranking terminé: {successful_count}/{len(documents)} documents traités")
        
        if successful_count > 0:
            valid_scores = [s for s in scores if s != 0.5]
            if valid_scores:
                top_scores = sorted(valid_scores, reverse=True)[:3]
                print(f"  📈 Top 3 scores: {[f'{s:.3f}' for s in top_scores]}")
        
        return scores
    
    def get_parameter_count(self) -> float:
        """Retourne le nombre de paramètres du modèle en milliards"""
        if self.model is None:
            return 0.0
        try:
            return sum(p.numel() for p in self.model.parameters()) / 1e9
        except:
            return 0.0
    
    def is_available(self) -> bool:
        """Vérifie si le reranker est disponible et fonctionnel"""
        return self.model is not None and self.tokenizer is not None


class GenericRAGChatbot:
    """Chatbot RAG générique utilisant les embeddings de Step 02 et Qwen3-4B-Instruct pour la génération"""
    
    def __init__(self, 
                 generation_model: str = "Qwen/Qwen3-4B-Instruct-2507",
                 initial_k: int = 20,
                 final_k: int = 3,
                 use_flash_attention: bool = True,
                 use_reranker: bool = True):
        """
        Initialise le système RAG générique
        
        Args:
            generation_model: Modèle Qwen3 pour la génération
            initial_k: Nombre de candidats pour la recherche initiale
            final_k: Nombre de documents finaux après reranking
            use_flash_attention: Utiliser Flash Attention (désactivé automatiquement sur Mac)
            use_reranker: Utiliser le reranking Qwen3
        """
        self.generation_model_name = generation_model
        self.initial_k = initial_k
        self.final_k = final_k
        self.use_flash_attention = use_flash_attention
        self.use_reranker = use_reranker
        
        # Détection de l'environnement (local + ZeroGPU)
        self.is_zerogpu = ZEROGPU_AVAILABLE and os.getenv("SPACE_ID") is not None
        self.is_mps = torch.backends.mps.is_available() and not self.is_zerogpu
        self.is_cuda = torch.cuda.is_available()
        
        # Configuration du device
        if self.is_mps:
            self.device = torch.device("mps")
        elif self.is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        if self.is_zerogpu:
            print("🚀 Environnement ZeroGPU détecté - optimisations cloud")
            self.use_flash_attention = True  # ZeroGPU supporte Flash Attention
        elif self.is_mps and use_flash_attention:
            print("🍎 Mac avec MPS détecté - désactivation automatique de Flash Attention")
            self.use_flash_attention = False
        
        # Chargement des composants
        self._load_step03_config()
        self._load_embeddings_from_hf()
        self._load_embedding_model()
        self._load_reranker()
        self._load_generation_model()

    def _load_step03_config(self):
        """Charge la configuration Step 03"""
        try:
            self.config = Step03Config()
            print(f"✅ Configuration Step 03 chargée")
            print(f"  📦 Repository HF: {self.config.repo_id}")
            print(f"  📊 Embeddings: {self.config.total_vectors:,} vecteurs")
            print(f"  📏 Dimension: {self.config.vector_dimension}")
        except Exception as e:
            print(f"❌ Erreur de chargement de la configuration: {e}")
            raise

    def _load_embeddings_from_hf(self):
        """Télécharge et charge les embeddings depuis Hugging Face Hub"""
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            import numpy as np
            import faiss
            
            print(f"🔄 Téléchargement des embeddings depuis {self.config.repo_id}...")
            
            # Télécharger les fichiers (sans token pour les repos publics)
            try:
                embeddings_file = hf_hub_download(
                    repo_id=self.config.repo_id,
                    filename=self.config.embeddings_file,
                    repo_type="dataset",
                    token=None  # Forcer l'accès sans token pour les repos publics
                )
                
                metadata_file = hf_hub_download(
                    repo_id=self.config.repo_id,
                    filename=self.config.metadata_file,
                    repo_type="dataset",
                    token=None  # Forcer l'accès sans token pour les repos publics
                )
            except Exception as auth_error:
                print(f"  ⚠️ Erreur d'authentification: {auth_error}")
                print("  🔑 Essai avec token depuis les variables d'environnement...")
                
                # Essayer avec le token d'environnement
                import os
                hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
                
                if hf_token:
                    print("  🔑 Token trouvé, nouvel essai...")
                    embeddings_file = hf_hub_download(
                        repo_id=self.config.repo_id,
                        filename=self.config.embeddings_file,
                        repo_type="dataset",
                        token=hf_token
                    )
                    
                    metadata_file = hf_hub_download(
                        repo_id=self.config.repo_id,
                        filename=self.config.metadata_file,
                        repo_type="dataset",
                        token=hf_token
                    )
                else:
                    print("  ❌ Aucun token trouvé dans les variables d'environnement")
                    print("  💡 Solutions possibles:")
                    print("     1. Vérifiez que le repository est bien public")
                    print("     2. Connectez-vous avec: huggingface-cli login")
                    print("     3. Définissez HF_TOKEN dans les variables d'environnement")
                    raise auth_error
            
            print("  📥 Chargement des embeddings SafeTensors...")
            tensors = load_file(embeddings_file)
            embeddings_tensor = tensors["embeddings"]
            embeddings_np = embeddings_tensor.numpy().astype(np.float32)
            
            print("  📋 Chargement des métadonnées...")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Créer l'index FAISS (optimisé pour Mac)
            print("  🔧 Création de l'index FAISS...")
            dimension = embeddings_np.shape[1]
            
            # Configuration d'index FAISS selon l'environnement
            if self.is_zerogpu:
                print("  🚀 Index FAISS optimisé pour ZeroGPU (IndexHNSWFlat)")
                # Index sophistiqué pour ZeroGPU avec GPU puissant
                self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
                self.faiss_index.hnsw.efConstruction = 200
                self.faiss_index.hnsw.efSearch = 50
            elif self.is_mps:
                print("  🍎 Index FAISS optimisé pour Mac (IndexFlatIP)")
                # Index simple mais efficace sur Mac
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (plus stable sur Mac)
            else:
                print("  🐧 Index FAISS HNSW pour Linux/Windows")
                # Index plus sophistiqué pour autres plateformes
                self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
                self.faiss_index.hnsw.efConstruction = 200
                self.faiss_index.hnsw.efSearch = 50
            
            # Normaliser les embeddings pour IndexFlatIP (équivalent à cosine similarity)
            if self.is_mps:
                # Normalisation L2 pour que IndexFlatIP = cosine similarity
                norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                embeddings_np = embeddings_np / (norms + 1e-8)  # Éviter division par 0
            
            print(f"  📊 Ajout de {embeddings_np.shape[0]:,} vecteurs à l'index...")
            # Ajouter les vecteurs à l'index
            self.faiss_index.add(embeddings_np)
            
            # Récupérer les mappings et métadonnées de contenu
            self.ordered_ids = self.metadata.get('ordered_ids', [])
            self.id_to_idx = self.metadata.get('id_to_idx', {})
            self.content_metadata = self.metadata.get('content_metadata', {})
            
            
            print(f"✅ Embeddings chargés: {embeddings_np.shape[0]:,} vecteurs de dimension {dimension}")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des embeddings: {e}")
            raise

    def _load_embedding_model(self):
        """Charge le modèle d'embeddings pour les requêtes"""
        print(f"🔄 Chargement du modèle d'embeddings {self.config.embedding_model}...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            if self.use_flash_attention and self.is_cuda:
                print("  - Configuration avec Flash Attention 2 activée (CUDA)")
                try:
                    self.embedding_model = SentenceTransformer(
                        self.config.embedding_model,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2", 
                            "device_map": "auto"
                        },
                        tokenizer_kwargs={"padding_side": "left"}
                    )
                except Exception as flash_error:
                    print(f"  - Flash Attention échoué: {flash_error}")
                    print("  - Fallback vers configuration standard")
                    self.embedding_model = SentenceTransformer(self.config.embedding_model)
                    self.use_flash_attention = False
            else:
                print("  - Configuration standard (MPS/CPU ou Flash Attention désactivé)")
                model_kwargs = {}
                
                if self.is_mps:
                    model_kwargs = {"torch_dtype": torch.float32}
                
                if model_kwargs:
                    self.embedding_model = SentenceTransformer(
                        self.config.embedding_model,
                        model_kwargs=model_kwargs,
                        tokenizer_kwargs={"padding_side": "left"}
                    )
                else:
                    self.embedding_model = SentenceTransformer(self.config.embedding_model)
                
            print(f"✅ Modèle d'embeddings {self.config.embedding_model} chargé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur avec {self.config.embedding_model}: {e}")
            print("🔄 Fallback vers le modèle multilingual MiniLM...")
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.use_flash_attention = False

    def _load_reranker(self):
        """Charge le reranker Qwen3-Reranker-4B"""
        if self.use_reranker:
            try:
                effective_flash_attention = self.use_flash_attention and not self.is_mps
                self.reranker = Qwen3Reranker(use_flash_attention=effective_flash_attention)
            except Exception as e:
                print(f"❌ Erreur lors du chargement du reranker: {e}")
                print("🔄 Désactivation du reranking")
                self.use_reranker = False
                self.reranker = None
        else:
            self.reranker = None
            print("⚠️ Reranking désactivé par configuration")

    def _load_generation_model(self):
        """Charge le modèle de génération Qwen3-4B-Instruct"""
        print(f"🔄 Chargement du modèle de génération {self.generation_model_name}...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Chargement du tokenizer
            print("  - Chargement du tokenizer...")
            self.generation_tokenizer = AutoTokenizer.from_pretrained(self.generation_model_name)
            
            # Configuration du modèle selon la plateforme
            model_kwargs = self._get_generation_model_config()
            
            # Chargement du modèle
            print("  - Chargement du modèle...")
            self.generation_model = AutoModelForCausalLM.from_pretrained(
                self.generation_model_name,
                **model_kwargs
            )
            
            # Configuration du device
            self._setup_generation_device()
            
            print(f"✅ Modèle de génération chargé sur {self.generation_device}")
            print(f"  - Paramètres: {self._get_generation_parameter_count():.1f}B")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle de génération: {e}")
            print("💡 La génération sera désactivée")
            self.generation_model = None
            self.generation_tokenizer = None
            self.generation_device = None

    def _get_generation_model_config(self) -> Dict:
        """Retourne la configuration du modèle de génération selon la plateforme"""
        config = {}
        
        if self.is_mps:
            config["torch_dtype"] = torch.float32
            config["device_map"] = None
        elif self.is_cuda:
            config["torch_dtype"] = torch.float16
            if self.use_flash_attention:
                try:
                    config["attn_implementation"] = "flash_attention_2"
                    print("  - Flash Attention 2 activée pour génération")
                except Exception:
                    print("  - Flash Attention 2 non disponible pour génération")
            config["device_map"] = "auto"
        else:
            config["torch_dtype"] = torch.float32
            config["device_map"] = "cpu"
        
        return config

    def _setup_generation_device(self):
        """Configure le device pour le modèle de génération"""
        if self.is_mps:
            self.generation_device = torch.device("mps")
            self.generation_model = self.generation_model.to(self.generation_device)
        elif self.is_cuda:
            if hasattr(self.generation_model, 'device'):
                self.generation_device = next(self.generation_model.parameters()).device
            else:
                self.generation_device = torch.device("cuda")
                self.generation_model = self.generation_model.to(self.generation_device)
        else:
            self.generation_device = torch.device("cpu")
            self.generation_model = self.generation_model.to(self.generation_device)

    def _get_generation_parameter_count(self) -> float:
        """Retourne le nombre de paramètres du modèle de génération en milliards"""
        if self.generation_model is None:
            return 0.0
        try:
            return sum(p.numel() for p in self.generation_model.parameters()) / 1e9
        except:
            return 0.0

    def search_documents(self, query: str, final_k: int = None, use_reranking: bool = None) -> List[Dict]:
        """
        Recherche avancée avec reranking en deux étapes
        """
        k = final_k if final_k is not None else self.final_k
        initial_k = max(self.initial_k, k * 3)
        should_rerank = use_reranking if use_reranking is not None else self.use_reranker
        
        print(f"🔍 Recherche en deux étapes: {initial_k} candidats → reranking → {k} finaux")
        
        # Étape 1: Recherche par embedding avec FAISS
        if hasattr(self.embedding_model, 'prompts') and 'query' in self.embedding_model.prompts:
            query_embedding = self.embedding_model.encode([query], prompt_name="query")[0]
        else:
            query_embedding = self.embedding_model.encode([query])[0]
        
        # Recherche dans l'index FAISS
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Normaliser la requête sur Mac pour IndexFlatIP (consistency avec les embeddings)
        if self.is_mps:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
        
        distances, indices = self.faiss_index.search(query_vector, initial_k)
        
        if len(indices[0]) == 0:
            print("❌ Aucun document trouvé")
            return []
        
        print(f"📋 {len(indices[0])} candidats récupérés")
        
        # Conversion en format intermédiaire
        initial_results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.ordered_ids):
                doc_id = self.ordered_ids[idx]
                doc_metadata = self.content_metadata.get(doc_id, {})
                
                # Ajustement des scores selon le type d'index
                if self.is_mps:
                    # Sur Mac avec IndexFlatIP : distance = inner product (plus haut = plus similaire)
                    embedding_score = float(distance)  # Inner product normalisé = cosine similarity
                    embedding_distance = 1.0 - embedding_score  # Conversion en distance pour compatibilité
                else:
                    # Sur autres plateformes avec IndexHNSWFlat : distance euclidienne
                    embedding_distance = float(distance)
                    embedding_score = 1 - embedding_distance
                
                doc = {
                    'content': doc_metadata.get('chunk_content', 'Contenu non disponible'),
                    'metadata': doc_metadata,
                    'embedding_distance': embedding_distance,
                    'embedding_score': embedding_score,
                    'source': doc_metadata.get('source_file', 'Inconnu'),
                    'title': doc_metadata.get('title', 'Sans titre'),
                    'heading': doc_metadata.get('heading', ''),
                    'initial_rank': i + 1
                }
                initial_results.append(doc)
        
        # Étape 2: Reranking si disponible
        if should_rerank and self.reranker and self.reranker.model is not None:
            print("🎯 Application du reranking Qwen3...")
            
            documents = [doc['content'] for doc in initial_results]
            
            
            rerank_scores = self.reranker.rerank(query, documents)
            
            # Ajout des scores de reranking
            for doc, rerank_score in zip(initial_results, rerank_scores):
                doc['rerank_score'] = float(rerank_score)
            
            # Tri par score de reranking
            initial_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Mise à jour des positions finales
            for i, doc in enumerate(initial_results):
                doc['final_rank'] = i + 1
            
            print(f"✅ Reranking appliqué, top 5 scores: {[f'{doc['rerank_score']:.3f}' for doc in initial_results[:5]]}")
        else:
            print("⚠️ Reranking désactivé, utilisation des scores d'embedding uniquement")
            for doc in initial_results:
                doc['rerank_score'] = doc['embedding_score']
                doc['final_rank'] = doc['initial_rank']
        
        # Retour des top-k résultats finaux
        final_results = initial_results[:k]
        print(f"📊 {len(final_results)} documents finaux sélectionnés")
        
        return final_results

    @spaces.GPU(duration=120)  # ZeroGPU: alloue GPU pour 120s max pour génération
    def generate_response_stream(self, query: str, context: str, history: List = None):
        """
        Génère une réponse streamée basée sur le contexte et l'historique
        """
        if self.generation_model is None or self.generation_tokenizer is None:
            yield "❌ Modèle de génération non disponible"
            return
        
        # Construction du prompt système
        system_prompt = """Tu es un assistant expert qui répond aux questions en te basant uniquement sur les documents fournis dans le contexte.

Instructions importantes:
- Réponds en français de manière claire et précise
- Base-toi uniquement sur les informations du contexte fourni
- Si l'information n'est pas dans le contexte, dis-le clairement
- Utilise un ton professionnel adapté au domaine
- Structure ta réponse avec des paragraphes clairs"""
        
        # Construire le prompt complet
        messages = [{"role": "system", "content": system_prompt}]
        
        # Ajouter l'historique si fourni
        if history:
            for msg in history:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    messages.append({"role": msg.role, "content": msg.content})
        
        # Ajouter le contexte et la question
        user_message = f"Contexte:\n{context}\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Tokenisation
            inputs = self.generation_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Génération streamée
            from transformers import TextIteratorStreamer
            import threading
            
            streamer = TextIteratorStreamer(
                self.generation_tokenizer,
                timeout=10.0,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs = {
                "input_ids": inputs,
                "streamer": streamer,
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": self.generation_tokenizer.eos_token_id,
                "eos_token_id": self.generation_tokenizer.eos_token_id,
            }
            
            # Lancer la génération dans un thread séparé
            thread = threading.Thread(target=self.generation_model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Streamer les tokens
            for new_token in streamer:
                yield new_token
            
            thread.join()
            
        except Exception as e:
            yield f"❌ Erreur lors de la génération: {str(e)}"

    @spaces.GPU(duration=120)  # ZeroGPU: alloue GPU pour 120s max pour génération  
    def generate_response(self, query: str, context: str, history: List = None) -> str:
        """
        Génère une réponse basée sur le contexte et l'historique
        """
        if self.generation_model is None or self.generation_tokenizer is None:
            return "❌ Modèle de génération non disponible"
        
        # Construction du prompt système
        system_prompt = """Tu es un assistant expert qui répond aux questions en te basant uniquement sur les documents fournis dans le contexte.

Instructions importantes:
- Réponds en français de manière claire et précise
- Base-toi uniquement sur les informations du contexte fourni
- Si l'information n'est pas dans le contexte, dis-le clairement
- Utilise un ton professionnel adapté au domaine
- Structure ta réponse avec des paragraphes clairs"""
        
        # Construire le prompt complet
        messages = [{"role": "system", "content": system_prompt}]
        
        # Ajouter l'historique si fourni
        if history:
            for msg in history:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    if msg.role in ["user", "assistant"] and not getattr(msg, 'metadata', None):
                        messages.append({"role": msg.role, "content": msg.content})
        
        # Ajouter la question courante avec le contexte
        user_prompt = f"""Contexte documentaire:
{context}

Question: {query}

Réponds à cette question en te basant sur le contexte fourni."""
        
        messages.append({"role": "user", "content": user_prompt})
        
        # Formatage pour le modèle
        try:
            # Appliquer le template de chat du modèle
            formatted_prompt = self.generation_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenisation
            inputs = self.generation_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )
            
            # Déplacement vers le device
            inputs = {k: v.to(self.generation_device) for k, v in inputs.items()}
            
            # Génération
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.generation_tokenizer.eos_token_id,
                    eos_token_id=self.generation_tokenizer.eos_token_id,
                )
            
            # Décodage de la réponse
            full_response = self.generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraire seulement la nouvelle génération
            response = full_response[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération: {e}")
            return f"❌ Erreur lors de la génération de la réponse: {str(e)}"

    def stream_response_with_tools(self, query: str, history, top_k: int = None, use_reranking: bool = None):
        """
        Génère une réponse streamée avec affichage visuel des tools et reranking Qwen3
        """
        # 1. S'assurer que l'historique est une liste
        if not history:
            history = []
        
        # 2. Ajouter le message utilisateur seulement s'il n'est pas déjà présent
        if not history or history[-1].role != "user" or history[-1].content != query:
            history.append(ChatMessage(role="user", content=query))
            yield history
            time.sleep(0.1)
        
        # 3. Recherche des documents avec tool visuel
        should_rerank = use_reranking if use_reranking is not None else self.use_reranker
        search_method = "avec reranking Qwen3" if should_rerank else "par embedding seulement"
        
        history.append(ChatMessage(
            role="assistant",
            content=f"Je recherche les documents les plus pertinents dans la base de données ({search_method})...",
            metadata={"title": "🔍 Recherche sémantique avancée"}
        ))
        yield history
        
        # Recherche des documents pertinents
        relevant_docs = self.search_documents(query, top_k, use_reranking)
        
        time.sleep(0.2)
        
        if not relevant_docs:
            history.append(ChatMessage(
                role="assistant",
                content="Aucun document pertinent trouvé dans la base de données."
            ))
            yield history
            return
        
        # 4. Affichage des documents trouvés avec scores détaillés
        docs_summary = f"Trouvé {len(relevant_docs)} documents pertinents"
        if should_rerank:
            docs_summary += f"\n\n📊 **Reranking Qwen3 appliqué:**"
            for i, doc in enumerate(relevant_docs):
                embedding_score = doc.get('embedding_score', 0)
                rerank_score = doc.get('rerank_score', 0)
                rank_change = doc.get('initial_rank', i+1) - doc.get('final_rank', i+1)
                rank_indicator = f" (#{doc.get('initial_rank', i+1)}→#{doc.get('final_rank', i+1)})" if rank_change != 0 else ""
                docs_summary += f"\n• **{doc['title']}**{rank_indicator}"
                docs_summary += f"\n  └ Embedding: {embedding_score:.3f} | Reranking: {rerank_score:.3f}"
        else:
            for i, doc in enumerate(relevant_docs):
                embedding_score = doc.get('embedding_score', doc.get('distance', 0))
                docs_summary += f"\n• **{doc['title']}** - Score: {embedding_score:.3f}"
        
        history.append(ChatMessage(
            role="assistant",
            content=docs_summary,
            metadata={"title": f"📚 Documents sélectionnés ({len(relevant_docs)} total)"}
        ))
        yield history
        
        time.sleep(0.2)
        
        # 5. Construction du contexte
        context_parts = []
        sources_with_scores = []

        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"[Document {i+1}] {doc['title']} - {doc['heading']}\n{doc['content']}")
            sources_with_scores.append({
                'title': doc['title'],
                'source': doc['source'],
                'embedding_score': doc.get('embedding_score', 1 - doc.get('distance', 0)),
                'rerank_score': doc.get('rerank_score'),
                'final_rank': doc.get('final_rank', i+1)
            })

        context = "\n\n".join(context_parts)
        
        # 6. Génération de la réponse avec Qwen3-4B
        history.append(ChatMessage(
            role="assistant",
            content="Génération de la réponse basée sur les documents sélectionnés...",
            metadata={"title": "🤖 Génération avec Qwen3-4B"}
        ))
        yield history
        
        time.sleep(0.2)
        
        # Génération streamée de la réponse
        history.append(ChatMessage(
            role="assistant", 
            content="",  # Commencer avec un contenu vide
            metadata={"title": "🤖 Réponse générée"}
        ))
        
        # Streamer la réponse token par token
        current_response = ""
        for token in self.generate_response_stream(query, context, history[:-1]):  # Exclure le dernier message vide
            current_response += token
            # Mettre à jour le dernier message avec la réponse en cours
            history[-1] = ChatMessage(
                role="assistant",
                content=current_response,
                metadata={"title": "🤖 Réponse générée"}
            )
            yield history
            time.sleep(0.01)  # Petit délai pour un streaming fluide
        
        time.sleep(0.2)
        
        # 7. Ajout des sources consultées avec scores détaillés
        sources_text = []
        for i, source_info in enumerate(sources_with_scores):
            embedding_score = source_info['embedding_score']
            rerank_score = source_info.get('rerank_score')
            source_file = source_info['source']
            
            if rerank_score is not None:
                score_display = f"Embedding: {embedding_score:.3f} | **Reranking: {rerank_score:.3f}**"
            else:
                score_display = f"Score: {embedding_score:.3f}"
            
            sources_text.append(f"• **[{i+1}]** {source_info['title']} ({source_file})\n  └ {score_display}")
        
        sources_display = "\n".join(sources_text)
        
        # Titre adaptatif selon la méthode utilisée
        sources_title = f"📚 Sources avec reranking Qwen3 ({len(relevant_docs)} documents)" if should_rerank else f"📚 Sources par embedding ({len(relevant_docs)} documents)"
        
        history.append(ChatMessage(
            role="assistant",
            content=sources_display,
            metadata={"title": sources_title}
        ))
        yield history


def _create_rag_system():
    """Créé et configure le système RAG avec paramètres optimaux"""
    
    # Détection automatique d'environnement
    is_zerogpu = ZEROGPU_AVAILABLE and os.getenv("SPACE_ID") is not None
    is_mac = torch.backends.mps.is_available() and not is_zerogpu
    is_cuda = torch.cuda.is_available()
    
    if is_zerogpu:
        print("🚀 ZeroGPU détecté - optimisations cloud appliquées")
    elif is_mac:
        print("🍎 Mac avec MPS détecté - optimisations automatiques appliquées")
    elif is_cuda:
        print("🐧 CUDA détecté - optimisations GPU appliquées")
    else:
        print("💻 CPU détecté - optimisations processeur appliquées")
    
    # Paramètres par défaut optimisés selon l'environnement
    if is_zerogpu:
        default_config = {
            'use_flash_attention': True,   # ZeroGPU supporte Flash Attention
            'use_reranker': True,          # GPU puissant, reranking activé
            'initial_k': 30,               # Plus de candidats avec GPU puissant
            'final_k': 5                   # Plus de documents finaux
        }
    elif is_mac:
        default_config = {
            'use_flash_attention': False,  # MPS ne supporte pas Flash Attention
            'use_reranker': True,          # Reranking OK sur Mac
            'initial_k': 20,               # Valeurs modérées
            'final_k': 3
        }
    else:
        default_config = {
            'use_flash_attention': is_cuda,  # Flash Attention seulement sur CUDA
            'use_reranker': True,            # Reranking par défaut
            'initial_k': 20,                 # Candidats pour la première étape
            'final_k': 3                     # Documents finaux par défaut
        }
    
    print("🚀 Initialisation du chatbot RAG générique...")
    return GenericRAGChatbot(**default_config)


def _clear_message():
    """Fonction utilitaire interne pour effacer le message d'entrée."""
    return ""

def _clear_chat():
    """Fonction utilitaire interne pour effacer l'historique de chat."""
    return []

def _ensure_chatmessages(history):
    """Convertit une liste en objets ChatMessage si besoin."""
    result = []
    for m in history or []:
        if isinstance(m, ChatMessage):
            result.append(m)
        elif isinstance(m, dict):
            result.append(ChatMessage(
                role=m.get("role", ""),
                content=m.get("content", ""),
                metadata=m.get("metadata", None)
            ))
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            result.append(ChatMessage(role=m[0], content=m[1]))
    return result


def chat_with_generic_rag(message, history, top_k, use_reranking):
    """
    Interface entre Gradio et le système RAG générique avec contrôles avancés.
    
    Cette fonction gère l'interface de chat interactive avec streaming en temps réel
    et affichage des étapes de traitement (recherche, reranking, génération).
    
    Args:
        message (str): Le message ou question de l'utilisateur à traiter
        history (list): L'historique de la conversation sous forme de liste de messages
        top_k (int): Nombre de documents finaux à utiliser pour la génération de réponse
        use_reranking (bool): Activation du reranking Qwen3 pour améliorer la sélection
        
    Yields:
        list: Historique mis à jour avec les nouveaux messages et étapes de traitement
    """
    history = _ensure_chatmessages(history)
    response_generator = rag_system.stream_response_with_tools(message, history, top_k, use_reranking)
    for updated_history in response_generator:
        yield updated_history


def ask_rag_question(question: str = "Qu'est-ce que Swift MLX?", num_documents: int = 3, use_reranking: bool = True) -> str:
    """
    Pose une question au système RAG LocalRAG et retourne la réponse avec les documents sources.
    
    Cette fonction utilise un système de recherche sémantique avancé avec des modèles Qwen3
    pour interroger une base de connaissances et générer des réponses contextualisées.

    Args:
        question (str): La question à poser au système RAG en langage naturel
        num_documents (int): Nombre de documents à utiliser pour générer la réponse (entre 1 et 10)
        use_reranking (bool): Utiliser le reranking Qwen3-Reranker-4B pour améliorer la sélection des documents

    Returns:
        str: Réponse générée incluant la réponse contextuelle et les sources avec leurs scores de pertinence
    """
    global rag_system
    
    try:
        # Validation des paramètres
        num_documents = max(1, min(10, int(num_documents)))
        
        print(f"🔍 Question MCP: {question}")
        print(f"📊 Paramètres: {num_documents} documents, reranking: {use_reranking}")
        
        # Recherche des documents pertinents
        relevant_docs = rag_system.search_documents(question, num_documents, use_reranking)
        
        if not relevant_docs:
            return "❌ Aucun document pertinent trouvé dans la base de données pour répondre à cette question."
        
        # Construction du contexte pour la génération
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"[Document {i+1}] {doc['title']} - {doc['heading']}\n{doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Génération de la réponse
        response = rag_system.generate_response(question, context, None)
        
        # Formatage de la réponse avec les sources
        sources_info = []
        search_method = "avec reranking Qwen3" if use_reranking else "par embedding seulement"
        
        sources_info.append(f"\n\n📚 **Documents sources utilisés ({search_method}):**\n")
        
        for i, doc in enumerate(relevant_docs):
            embedding_score = doc.get('embedding_score', 0)
            rerank_score = doc.get('rerank_score')
            initial_rank = doc.get('initial_rank', i+1)
            final_rank = doc.get('final_rank', i+1)
            
            # Formatage des scores
            if rerank_score is not None and use_reranking:
                score_display = f"Embedding: {embedding_score:.3f} | **Reranking: {rerank_score:.3f}**"
                if initial_rank != final_rank:
                    rank_change = f" (#{initial_rank}→#{final_rank})"
                else:
                    rank_change = ""
            else:
                score_display = f"Score: {embedding_score:.3f}"
                rank_change = ""
                
            sources_info.append(f"• **[{i+1}]** {doc['title']}{rank_change}")
            sources_info.append(f"  └ {score_display}")
            sources_info.append(f"  └ Source: {doc['source']}")
        
        # Assemblage de la réponse finale
        final_response = response + "\n".join(sources_info)
        
        print(f"✅ Réponse MCP générée ({len(relevant_docs)} documents utilisés)")
        return final_response
        
    except Exception as e:
        error_msg = f"❌ Erreur lors du traitement de la question: {str(e)}"
        print(error_msg)
        return error_msg


def main():
    """Point d'entrée principal."""
    print("🚀 LocalRAG Step 03 - Interface de chat générique")
    print("=" * 50)
    
    # Vérification des dépendances
    if not _check_dependencies():
        return 1
    
    # Initialisation du système RAG
    global rag_system
    try:
        rag_system = _create_rag_system()
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return 1
    
    # Configuration de l'interface Gradio avec thème Glass
    with gr.Blocks(
        title="🤖 LocalRAG Chat Générique",
        theme=gr.themes.Glass(),
    ) as demo:
        
        # En-tête simplifié avec composants Gradio natifs
        with gr.Row():
            with gr.Column():
                gr.Markdown("# 🤖 Assistant RAG Générique LocalRAG")
                
                # Affichage de l'environnement d'exécution
                env_info = ""
                if ZEROGPU_AVAILABLE and os.getenv("SPACE_ID"):
                    env_info = "🚀 **Powered by ZeroGPU** - GPU gratuit Hugging Face"
                elif torch.backends.mps.is_available():
                    env_info = "🍎 **Apple Silicon optimisé** - MPS accelerated"
                elif torch.cuda.is_available():
                    env_info = f"🐧 **CUDA accelerated** - {torch.cuda.get_device_name()}"
                else:
                    env_info = "💻 **CPU optimisé** - Traitement local"
                
                gr.Markdown(f"**Système RAG complet avec modèles Qwen3 de dernière génération**")
                gr.Markdown(env_info)
                gr.Markdown(f"🧠 {rag_system.config.embedding_model.split('/')[-1]} • 🎯 Qwen3-Reranker-4B • 💬 Qwen3-4B • ⚡ Recherche en 2 étapes")
                gr.Markdown(f"📦 Repository: `{rag_system.config.repo_id}` | 📊 Vecteurs: **{rag_system.config.total_vectors:,}**")
        
        # Interface de chat
        chatbot = gr.Chatbot(
            height=500,
            show_label=False,
            container=True,
            show_copy_button=True,
            autoscroll=True,
            avatar_images=(None, "🤖"),
            type="messages"
        )
        
        # Zone de saisie
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Posez votre question...",
                show_label=False,
                container=False,
                scale=4
            )
            send_btn = gr.Button("📤 Envoyer", variant="primary", scale=1)
        
        # Panneau de contrôle avancé simplifié
        with gr.Accordion("🎛️ Contrôles avancés", open=True):
            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="📊 Nombre de documents finaux",
                    info="Documents qui seront utilisés pour générer la réponse"
                )
                
                reranking_checkbox = gr.Checkbox(
                    value=True,
                    label="🎯 Activer le reranking Qwen3",
                    info="Améliore la pertinence avec un modèle de reranking spécialisé"
                )
        
        # Bouton pour effacer
        clear_btn = gr.Button("🗑️ Effacer la conversation", variant="secondary", size="lg")
        
        # Informations en pied de page avec Accordion pour économiser l'espace
        with gr.Accordion("ℹ️ Informations sur l'architecture", open=False):
            env_docs = ""
            if ZEROGPU_AVAILABLE and os.getenv("SPACE_ID"):
                env_docs = """
            ### 🚀 Optimisations ZeroGPU
            
            - **Allocation dynamique :** GPU alloué automatiquement pour le reranking et la génération
            - **NVIDIA H200 :** 70GB VRAM disponible pour les calculs intensifs
            - **Décorateurs intelligents :** `@spaces.GPU()` pour optimiser l'usage GPU
            - **Cache optimisé :** Stockage temporaire en `/tmp` pour performances maximales
            """
            elif torch.backends.mps.is_available():
                env_docs = """
            ### 🍎 Optimisations Apple Silicon
            
            - **Metal Performance Shaders :** Accélération native Apple
            - **Index FAISS adapté :** IndexFlatIP pour éviter les segfaults
            - **Mémoire unifiée :** Partage efficace CPU/GPU
            - **Float32 :** Précision optimisée pour MPS
            """
            else:
                env_docs = """
            ### ⚡ Optimisations locales
            
            - **Multi-plateforme :** Support CPU, CUDA, MPS selon disponibilité
            - **Flash Attention :** Activé automatiquement sur CUDA
            - **Gestion mémoire :** Cleanup automatique pour stabilité
            """
            
            gr.Markdown(f"""
            ### 🚀 Architecture LocalRAG Step 03
            
            - **📥 Step 02 :** Embeddings chargés depuis Hugging Face Hub au format SafeTensors
            - **🔍 Recherche :** Index FAISS reconstructé pour recherche vectorielle haute performance
            - **🎯 Reranking :** Qwen3-Reranker-4B pour affiner la sélection des documents
            - **💬 Génération :** Qwen3-4B-Instruct-2507 pour des réponses contextuelles optimisées
            {env_docs}
            ### 📊 Lecture des scores
            
            - **Score Embedding :** Similarité vectorielle initiale (0.0-1.0, plus haut = plus pertinent)
            - **Score Reranking :** Score de pertinence final après analyse contextuelle
            - **Changement de rang :** Evolution de la position du document après reranking
            """)

        # Gestionnaire de likes
        def like_response(evt: gr.LikeData):
            print(f"Réaction utilisateur: {'👍' if evt.liked else '👎'} sur le message #{evt.index}")
            print(f"Contenu: {evt.value[:100]}...")
        
        chatbot.like(like_response)
        
        # Envoi par touche Entrée
        msg.submit(
            chat_with_generic_rag,
            [msg, chatbot, top_k_slider, reranking_checkbox],
            chatbot
        ).then(
            _clear_message,
            outputs=msg
        )
        
        # Envoi par bouton
        send_btn.click(
            chat_with_generic_rag,
            [msg, chatbot, top_k_slider, reranking_checkbox],
            chatbot
        ).then(
            _clear_message,
            outputs=msg
        )
        
        # Effacement de la conversation
        clear_btn.click(_clear_chat, outputs=chatbot)

        print("🌐 Lancement de l'interface Gradio...")
        
        # Configuration HTTPS pour Claude Desktop
        ssl_keyfile = os.getenv("SSL_KEYFILE")
        ssl_certfile = os.getenv("SSL_CERTFILE")
        
        if ssl_keyfile and ssl_certfile:
            print("🔒 Mode HTTPS activé")
            print("🔗 Serveur MCP : /gradio_api/mcp/sse")
            
            demo.launch(
                mcp_server=True,  # Toujours activer MCP
                inbrowser=True,
                show_error=True,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile
            )
        else:
            print("🔗 Serveur MCP : /gradio_api/mcp/sse")
            print("💡 Pour HTTPS : python step03_ssl_generator_optional.py")
            
            demo.launch(
                mcp_server=True,  # Toujours activer MCP
                inbrowser=True,
                show_error=True
            )
        
        print("📋 Outil MCP exposé : ask_rag_question")

    return 0


if __name__ == "__main__":
    exit(main())