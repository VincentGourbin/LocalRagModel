#!/usr/bin/env python3
"""
Système d'indexation universelle pour documentation technique
Supporte HTML et Markdown avec analyse d'images intégrées
Utilise Qwen3-Embedding-4B + Qwen3-Reranker-4B + Ollama pour l'analyse multimodale et FAISS pour le stockage vectoriel
Version robuste avec gestion d'erreurs centralisée et système de tracking JSON simplifié
"""

import os
import re
import json
import hashlib
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from enum import Enum
from datetime import datetime

try:
    import faiss
    from faiss_indexer import FAISSIndexer
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️ FAISS non disponible - installez avec: pip install faiss-cpu")
    FAISS_AVAILABLE = False
from ollama import chat
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from PIL import Image
import requests
from tqdm import tqdm
import numpy as np
import markdown
from markdown.extensions import codehilite, toc, tables

class ErrorType(Enum):
    """Énumération des types d'erreurs possibles lors du traitement des documents.
    
    Permet de catégoriser et traiter différemment les erreurs selon leur nature,
    facilitant ainsi le débogage et l'affichage d'informations pertinentes.
    """
    IMAGE_NOT_FOUND = "image_not_found"
    IMAGE_NAME_TOO_LONG = "image_name_too_long"
    IMAGE_ANALYSIS_FAILED = "image_analysis_failed"
    PARSING_ERROR = "parsing_error"
    EMBEDDING_ERROR = "embedding_error"
    INVALID_HTML = "invalid_html"

@dataclass
class ProcessingError:
    """Représente une erreur survenue lors du traitement d'un document.
    
    Cette classe encapsule toutes les informations nécessaires pour
    identifier et diagnostiquer une erreur de traitement.
    
    Attributes:
        error_type: Type d'erreur selon l'énumération ErrorType
        file_path: Chemin du fichier où l'erreur s'est produite
        message: Message d'erreur principal
        details: Informations détaillées sur l'erreur (optionnel)
    """
    error_type: ErrorType
    file_path: str
    message: str
    details: str = ""

@dataclass
class DocumentChunk:
    """Représente un segment (chunk) de document avec ses métadonnées.
    
    Cette classe constitue l'unité de base pour l'indexation vectorielle.
    Chaque chunk contient un fragment de contenu textuel accompagné de ses
    métadonnées structurelles (titre, section) et multimédia (images, liens).
    
    Attributes:
        content: Contenu textuel du chunk
        source_file: Chemin du fichier source
        chunk_id: Identifiant unique du chunk
        title: Titre du document parent
        heading: En-tête de la section contenant le chunk
        images: Liste des informations sur les images présentes
        links: Liste des liens hypertextes trouvés
    """
    """Représente un chunk de document avec ses métadonnées"""
    content: str
    source_file: str
    chunk_id: str
    title: str = ""
    heading: str = ""
    images: List[Dict] = None
    links: List[str] = None
    
    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.links is None:
            self.links = []

@dataclass
class SearchResult:
    """Résultat d'une recherche sémantique avec scores de pertinence.
    
    Encapsule un chunk de document trouvé lors d'une recherche, accompagné
    des métriques de pertinence calculées par l'embedding model et le reranker.
    
    Attributes:
        chunk: Le chunk de document correspondant
        similarity_score: Score de similarité cosinus (0-1)
        rerank_score: Score du reranker si disponible (0-1)
        rank_position: Position dans le classement final
    """
    """Résultat de recherche avec score de reranking"""
    chunk: DocumentChunk
    similarity_score: float
    rerank_score: Optional[float] = None
    rank_position: int = 0

class ErrorCollector:
    """Collecteur centralisé d'erreurs avec affichage en temps réel.
    
    Cette classe permet de collecter et afficher les erreurs survenues
    lors du traitement des documents, avec possibilité d'affichage immédiat
    ou différé selon les besoins.
    
    Le système d'icônes et de catégorisation facilite l'identification
    rapide des types de problèmes rencontrés.
    
    Attributes:
        errors: Liste des erreurs collectées
        warnings: Liste des avertissements
        show_immediate: Si True, affiche les erreurs immédiatement
    """
    """Collecteur d'erreurs pour le traitement des documents avec affichage en temps réel"""
    
    def __init__(self, show_immediate: bool = True):
        """Initialise le collecteur d'erreurs.
        
        Args:
            show_immediate: Si True, affiche les erreurs dès qu'elles surviennent
        """
        self.errors: List[ProcessingError] = []
        self.warnings: List[str] = []
        self.show_immediate = show_immediate
        
    def add_error(self, error_type: ErrorType, file_path: str, message: str, details: str = ""):
        """Ajoute une erreur à la collection avec affichage optionnel immédiat.
        
        Args:
            error_type: Type d'erreur selon l'énumération ErrorType
            file_path: Chemin du fichier concerné
            message: Message d'erreur principal
            details: Informations complémentaires sur l'erreur
        """
        """Ajoute une erreur à la collection et l'affiche immédiatement si demandé"""
        error = ProcessingError(error_type, file_path, message, details)
        self.errors.append(error)
        
        if self.show_immediate:
            self._print_immediate_error(error)
        
    def add_warning(self, message: str):
        """Ajoute un avertissement avec affichage optionnel immédiat.
        
        Args:
            message: Message d'avertissement
        """
        """Ajoute un avertissement et l'affiche immédiatement si demandé"""
        self.warnings.append(message)
        
        if self.show_immediate:
            print(f"   ⚠️  AVERTISSEMENT: {message}")
    
    def _print_immediate_error(self, error: ProcessingError):
        """Affiche une erreur immédiatement avec formatage et icônes.
        
        Utilise un système d'icônes pour identifier visuellement
        le type d'erreur et faciliter le débogage.
        
        Args:
            error: L'erreur à afficher
        """
        """Affiche une erreur immédiatement"""
        error_icons = {
            ErrorType.IMAGE_NOT_FOUND: "🖼️❌",
            ErrorType.IMAGE_NAME_TOO_LONG: "📏❌", 
            ErrorType.IMAGE_ANALYSIS_FAILED: "🔍❌",
            ErrorType.PARSING_ERROR: "📄❌",
            ErrorType.EMBEDDING_ERROR: "🧠❌",
            ErrorType.INVALID_HTML: "🌐❌"
        }
        
        icon = error_icons.get(error.error_type, "❌")
        type_name = error.error_type.value.replace('_', ' ').title()
        
        print(f"   {icon} ERREUR [{type_name}]: {error.message}")
        if error.details:
            print(f"       Détails: {error.details}")
    
    def print_summary(self, show_if_immediate: bool = True):
        """Affiche un résumé complet des erreurs et avertissements.
        
        Propose deux modes d'affichage :
        - Compact si les erreurs ont déjà été affichées en temps réel
        - Détaillé sinon, avec regroupement par type d'erreur
        
        Args:
            show_if_immediate: Contrôle l'affichage si show_immediate est activé
        """
        """Affiche le résumé des erreurs en fin de traitement"""
        # Si on affiche déjà en temps réel, on peut faire un résumé plus compact
        if self.show_immediate and show_if_immediate:
            if not self.errors and not self.warnings:
                print("✅ Aucune erreur rencontrée")
                return
                
            print(f"\n📋 RÉSUMÉ FINAL:")
            print("=" * 40)
            
            if self.warnings:
                print(f"⚠️  Avertissements: {len(self.warnings)}")
            
            if self.errors:
                # Compter par type d'erreur
                error_counts = {}
                for error in self.errors:
                    error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
                
                print(f"❌ Erreurs: {len(self.errors)} total")
                for error_type, count in error_counts.items():
                    type_name = error_type.value.replace('_', ' ').title()
                    print(f"   • {type_name}: {count}")
            
            print("=" * 40)
            return
        
        # Affichage complet si pas d'affichage immédiat
        if not self.errors and not self.warnings:
            print("✅ Aucune erreur rencontrée")
            return
            
        print(f"\n⚠️  RÉSUMÉ DES ERREURS ET AVERTISSEMENTS")
        print("=" * 60)
        
        if self.warnings:
            print(f"\n📋 Avertissements ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.errors:
            print(f"\n❌ Erreurs ({len(self.errors)}):")
            
            # Grouper par type d'erreur
            errors_by_type = {}
            for error in self.errors:
                if error.error_type not in errors_by_type:
                    errors_by_type[error.error_type] = []
                errors_by_type[error.error_type].append(error)
            
            for error_type, errors in errors_by_type.items():
                print(f"\n   🔍 {error_type.value.replace('_', ' ').title()} ({len(errors)}):")
                for error in errors:
                    print(f"      📄 {os.path.basename(error.file_path)}")
                    print(f"         {error.message}")
                    if error.details:
                        print(f"         Détails: {error.details}")
        
        print("=" * 60)

class IndexingTracker:
    """Gestionnaire de suivi d'indexation utilisant un fichier JSON.
    
    Cette classe implémente un système de tracking léger pour éviter
    la réindexation inutile de fichiers non modifiés. Elle utilise
    un fichier JSON pour stocker les métadonnées des fichiers indexés.
    
    Le système compare les timestamps de modification (mtime) pour
    détecter les changements et optimiser les opérations d'indexation.
    
    Attributes:
        tracking_file: Chemin vers le fichier JSON de tracking
        tracking_data: Dictionnaire des données de suivi chargées
    """
    """Gestionnaire simple pour suivre l'état d'indexation des fichiers avec fichier JSON"""
    
    def __init__(self, db_path: str):
        self.tracking_file = Path(db_path) / "indexing_tracker.json"
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Charge les données de tracking depuis le fichier JSON.
        
        Gestion robuste des erreurs de lecture avec fallback
        vers un dictionnaire vide si le fichier est corrompu.
        
        Returns:
            Dict: Données de tracking ou dictionnaire vide si erreur
        """
        """Charge les données de tracking depuis le fichier JSON"""
        try:
            if self.tracking_file.exists():
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement du tracking: {e}")
            return {}
    
    def _save_tracking_data(self):
        """Sauvegarde les données de tracking dans le fichier JSON"""
        try:
            # Créer le répertoire si nécessaire
            self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracking_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Erreur lors de la sauvegarde du tracking: {e}")
    
    def get_file_info(self, file_path: Path) -> Optional[Dict]:
        """Récupère les informations d'un fichier depuis le tracking"""
        file_key = str(file_path.resolve())
        return self.tracking_data.get(file_key)
    
    def is_file_indexed(self, file_path: Path) -> bool:
        """Vérifie si un fichier a déjà été indexé.
        
        Args:
            file_path: Chemin vers le fichier à vérifier
            
        Returns:
            bool: True si le fichier est présent dans le tracking
        """
        """Vérifie si un fichier a déjà été indexé"""
        return self.get_file_info(file_path) is not None
    
    def needs_reindexing(self, file_path: Path) -> bool:
        """Détermine si un fichier a besoin d'être réindexé.
        
        Compare le timestamp de modification du fichier avec celui
        stocké dans le tracking. Retourne True si :
        - Le fichier n'a jamais été indexé
        - Le fichier a été modifié depuis la dernière indexation
        
        Args:
            file_path: Chemin vers le fichier à analyser
            
        Returns:
            bool: True si réindexation nécessaire, False sinon
        """
        """Vérifie si un fichier a besoin d'être réindexé"""
        try:
            file_info = self.get_file_info(file_path)
            
            if file_info is None:
                return True  # Fichier jamais indexé
            
            # Vérifier si le fichier existe encore
            if not file_path.exists():
                return False  # Fichier supprimé, pas besoin de réindexer
            
            # Comparer les timestamps
            current_mtime = file_path.stat().st_mtime
            stored_mtime = file_info.get('mtime', 0)
            
            return current_mtime > stored_mtime
            
        except Exception:
            return True  # En cas d'erreur, on réindexe par sécurité
    
    def mark_file_indexed(self, file_path: Path, chunks_count: int = 0, images_count: int = 0):
        """Marque un fichier comme indexé avec ses statistiques.
        
        Enregistre les métadonnées du fichier dans le tracking JSON :
        - Timestamp de modification (mtime)
        - Taille du fichier
        - Date d'indexation
        - Statistiques de traitement
        
        Args:
            file_path: Chemin du fichier traité
            chunks_count: Nombre de chunks créés
            images_count: Nombre d'images traitées
        """
        """Marque un fichier comme indexé avec ses métadonnées"""
        try:
            file_key = str(file_path.resolve())
            
            file_stats = file_path.stat()
            
            self.tracking_data[file_key] = {
                'mtime': file_stats.st_mtime,
                'size': file_stats.st_size,
                'indexed_at': datetime.now().isoformat(),
                'chunks_count': chunks_count,
                'images_count': images_count,
                'file_name': file_path.name
            }
            
            self._save_tracking_data()
            
        except Exception as e:
            print(f"⚠️  Erreur lors du marquage du fichier {file_path}: {e}")
    
    def remove_file_tracking(self, file_path: Path):
        """Supprime le tracking d'un fichier"""
        file_key = str(file_path.resolve())
        if file_key in self.tracking_data:
            del self.tracking_data[file_key]
            self._save_tracking_data()
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du tracking"""
        if not self.tracking_data:
            return {
                'total_files': 0,
                'total_chunks': 0,
                'total_images': 0,
                'oldest_index': None,
                'newest_index': None
            }
        
        total_chunks = sum(info.get('chunks_count', 0) for info in self.tracking_data.values())
        total_images = sum(info.get('images_count', 0) for info in self.tracking_data.values())
        
        dates = [info.get('indexed_at') for info in self.tracking_data.values() if info.get('indexed_at')]
        
        return {
            'total_files': len(self.tracking_data),
            'total_chunks': total_chunks,
            'total_images': total_images,
            'oldest_index': min(dates) if dates else None,
            'newest_index': max(dates) if dates else None
        }
    
    def cleanup_deleted_files(self, existing_files: List[Path]):
        """Nettoie le tracking des fichiers qui n'existent plus"""
        existing_keys = {str(f.resolve()) for f in existing_files}
        to_remove = []
        
        for file_key in self.tracking_data:
            if file_key not in existing_keys:
                to_remove.append(file_key)
        
        for file_key in to_remove:
            del self.tracking_data[file_key]
        
        if to_remove:
            print(f"🧹 Nettoyage: {len(to_remove)} fichiers supprimés du tracking")
            self._save_tracking_data()
        
        return len(to_remove)

class OllamaImageAnalyzer:
    """Analyseur d'images utilisant Ollama avec un modèle de vision multimodal.
    
    Cette classe permet d'analyser des images techniques de documentation
    et de générer des descriptions détaillées en français. Elle utilise
    le modèle qwen2.5vl:7b par défaut.
    
    L'analyseur est configuré pour comprendre :
    - Les schémas et diagrammes techniques
    - Les captures d'écran d'interfaces
    - Les graphiques et tableaux de données
    - Les diagrammes de flux et architectures
    - Les exemples de code avec annotations visuelles
    
    Attributes:
        model_name: Nom du modèle Ollama utilisé
    """
    """Analyseur d'images utilisant Ollama avec modèle de vision"""
    
    def __init__(self, model_name: str = "qwen2.5vl:7b"):
        self.model_name = model_name
        print(f"Configuration d'Ollama avec le modèle: {model_name}")
        
        # Test de connexion à Ollama
        try:
            test_response = chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'Test de connexion.'}],
            )
            print("✅ Ollama et modèle de vision disponibles")
        except Exception as e:
            print(f"❌ Erreur de connexion à Ollama: {e}")
            print("Vérifiez qu'Ollama est lancé et que le modèle est installé:")
            print(f"  ollama pull {model_name}")
            raise
    
    def analyze_image(self, image_path: str, context: str = "") -> str:
        """Analyse une image et génère sa description technique.
        
        Utilise un prompt spécialisé pour l'analyse d'images techniques
        du domaine de la construction et du béton. La description générée
        couvre les aspects visuels et techniques pertinents.
        
        Args:
            image_path: Chemin vers l'image à analyser
            context: Contexte additionnel pour guider l'analyse
            
        Returns:
            str: Description détaillée de l'image en français
        """
        """Analyse une image et retourne sa description"""
        try:
            if not os.path.exists(image_path):
                return f"Image non trouvée: {os.path.basename(image_path)}"
            
            base_prompt = "Analysez cette image technique de documentation."
            
            if context:
                base_prompt += f" Contexte: {context}."
            
            base_prompt += (" Décrivez précisément: 1) Ce qui est représenté (schéma, capture, graphique, diagramme), "
                          "2) Les éléments techniques visibles, 3) Le code ou les valeurs si présents, "
                          "4) L'objectif pédagogique ou informatif. Réponse concise et technique en français.")
            
            response = chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': base_prompt,
                    'images': [image_path],
                }],
                keep_alive=0
            )
            
            description = response.message.content.strip()
            
            if description and len(description) > 10:
                print(f"  📊 Description générée: {len(description)} caractères")
                return description
            else:
                return f"Image technique de documentation ({os.path.basename(image_path)})"
                
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de l'image {image_path}: {e}")
            return f"Image technique - erreur d'analyse ({os.path.basename(image_path)})"

class MarkdownDocumentParser:
    """Parser spécialisé pour documents Markdown avec analyse d'images.
    
    Cette classe implémente un parseur robuste capable de traiter :
    - Documents Markdown avec en-têtes hiérarchiques (H1-H6)
    - Blocs de code avec coloration syntaxique
    - Images intégrées avec analyse automatique du contenu
    - Tables et listes structurées
    - Liens internes et externes
    
    Le parser extrait le contenu sous forme de chunks sémantiquement
    cohérents basés sur la structure des en-têtes.
    
    Attributes:
        base_directory: Répertoire racine pour la résolution des chemins relatifs
        image_analyzer: Instance de l'analyseur d'images Ollama
        md: Instance du processeur Markdown configuré
    """
    
    def __init__(self, base_directory: str, image_analyzer: OllamaImageAnalyzer = None):
        self.base_directory = Path(base_directory)
        self.image_analyzer = image_analyzer or OllamaImageAnalyzer()

        # Configuration du processeur Markdown avec extensions
        self.md = markdown.Markdown(extensions=[
            'toc',           # Table des matières
            'tables',        # Support des tables
            'codehilite',    # Coloration syntaxique
            'fenced_code',   # Blocs de code avec ``` 
            'attr_list',     # Attributs pour les éléments
            'def_list',      # Listes de définitions
            'footnotes'      # Notes de bas de page
        ])
    
    def parse_markdown_file(self, file_path: Path, error_collector: ErrorCollector = None) -> List[DocumentChunk]:
        """Parse un fichier Markdown et extrait les chunks de contenu.
        
        Point d'entrée principal pour le traitement d'un document Markdown.
        Gestion robuste des erreurs avec collecte centralisée.
        
        Args:
            file_path: Chemin vers le fichier Markdown à traiter
            error_collector: Collecteur d'erreurs optionnel
            
        Returns:
            List[DocumentChunk]: Liste des chunks extraits du document
        """
        if error_collector is None:
            error_collector = ErrorCollector()
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extraction du titre depuis les métadonnées ou premier H1
            title = self._extract_title_from_markdown(content)
            
            # Extraction des chunks par sections
            chunks = self._extract_markdown_sections(content, file_path, title, error_collector)
            
            return chunks
            
        except Exception as e:
            error_collector.add_error(
                ErrorType.PARSING_ERROR,
                str(file_path),
                f"Erreur lors du parsing du fichier Markdown",
                str(e)
            )
            return []
    
    def _extract_title_from_markdown(self, content: str) -> str:
        """Extrait le titre du document Markdown.
        
        Args:
            content: Contenu brut du fichier Markdown
            
        Returns:
            str: Titre du document
        """
        # Recherche du premier H1 dans le contenu original
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        return "Document technique"
    
    def _extract_markdown_sections(self, content: str, file_path: Path, doc_title: str, error_collector: ErrorCollector) -> List[DocumentChunk]:
        """Extrait les sections du document Markdown basées sur les en-têtes.
        
        Args:
            content: Contenu brut du fichier Markdown
            file_path: Chemin du fichier source
            doc_title: Titre du document
            error_collector: Collecteur d'erreurs
            
        Returns:
            List[DocumentChunk]: Chunks extraits par section
        """
        chunks = []
        lines = content.split('\n')
        current_section = {'title': '', 'content': [], 'level': 0, 'start_line': 0}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Détection des en-têtes (# ## ### etc.)
            if line_stripped.startswith('#'):
                # Sauvegarder la section précédente si elle existe
                if current_section['content'] or current_section['title']:
                    chunk = self._create_chunk_from_section(
                        current_section, file_path, doc_title, error_collector
                    )
                    if chunk:
                        chunks.append(chunk)
                
                # Commencer une nouvelle section
                level = len(line_stripped) - len(line_stripped.lstrip('#'))
                title = line_stripped.lstrip('#').strip()
                
                current_section = {
                    'title': title,
                    'content': [],
                    'level': level,
                    'start_line': i
                }
            else:
                # Ajouter la ligne au contenu de la section courante
                current_section['content'].append(line)
        
        # Traiter la dernière section
        if current_section['content'] or current_section['title']:
            chunk = self._create_chunk_from_section(
                current_section, file_path, doc_title, error_collector
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_section(self, section: dict, file_path: Path, doc_title: str, error_collector: ErrorCollector) -> Optional[DocumentChunk]:
        """Crée un chunk à partir d'une section Markdown.
        
        Args:
            section: Dictionnaire contenant les informations de la section
            file_path: Chemin du fichier source
            doc_title: Titre du document
            error_collector: Collecteur d'erreurs
            
        Returns:
            Optional[DocumentChunk]: Chunk créé ou None si trop petit
        """
        try:
            heading = section['title']
            content_lines = section['content']
            
            # Reconstruction du contenu
            full_content = '\n'.join(content_lines).strip()
            
            if not heading and not full_content:
                return None
            
            # Construction du contenu final avec en-tête
            if heading:
                final_content = f"{heading}\n\n{full_content}"
            else:
                final_content = full_content
            
            # Filtrage des chunks trop petits
            if len(final_content.strip()) < 50:
                return None
            
            # Extraction des images et liens
            images = self._extract_images_from_markdown(full_content, file_path, error_collector)
            links = self._extract_links_from_markdown(full_content)
            
            # Génération de l'ID unique
            chunk_id = hashlib.md5(f"{file_path}_{section['start_line']}_{heading}".encode()).hexdigest()
            
            return DocumentChunk(
                content=final_content,
                source_file=str(file_path),
                chunk_id=chunk_id,
                title=doc_title,
                heading=heading or "Contenu principal",
                images=images,
                links=links
            )
            
        except Exception as e:
            error_collector.add_error(
                ErrorType.PARSING_ERROR,
                str(file_path),
                f"Erreur lors de la création du chunk pour la section: {section.get('title', 'Sans titre')}",
                str(e)
            )
            return None
    
    def _extract_images_from_markdown(self, content: str, file_path: Path, error_collector: ErrorCollector) -> List[Dict]:
        """Extrait et analyse les images d'un contenu Markdown.
        
        Args:
            content: Contenu Markdown de la section
            file_path: Chemin du fichier source
            error_collector: Collecteur d'erreurs
            
        Returns:
            List[Dict]: Informations sur les images trouvées
        """
        images = []
        
        # Pattern pour les images Markdown: ![alt](src "title")
        import re
        img_pattern = r'!\[([^\]]*)\]\(([^\)]+)(?:\s+"([^"]+)")?\)'
        
        for match in re.finditer(img_pattern, content):
            alt_text = match.group(1) or ''
            src = match.group(2)
            title_text = match.group(3) or ''
            
            # Traitement similaire au parser HTML
            img_info = self._process_markdown_image(src, alt_text, title_text, file_path, error_collector)
            if img_info:
                images.append(img_info)
        
        return images
    
    def _extract_links_from_markdown(self, content: str) -> List[str]:
        """Extrait les liens d'un contenu Markdown.
        
        Args:
            content: Contenu Markdown de la section
            
        Returns:
            List[str]: Liste des URLs trouvées
        """
        import re
        links = []
        
        # Pattern pour les liens Markdown: [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        
        for match in re.finditer(link_pattern, content):
            url = match.group(2)
            if not url.startswith('#'):  # Ignorer les ancres internes
                links.append(url)
        
        return links
    
    def _process_markdown_image(self, src: str, alt_text: str, title_text: str, file_path: Path, error_collector: ErrorCollector) -> Optional[Dict]:
        """Traite une image trouvée dans le Markdown.
        
        Logic similaire au traitement des images HTML mais adaptée
        pour les chemins relatifs typiques du Markdown.
        
        Args:
            src: Source de l'image
            alt_text: Texte alternatif
            title_text: Titre de l'image
            file_path: Chemin du fichier source
            error_collector: Collecteur d'erreurs
            
        Returns:
            Optional[Dict]: Informations sur l'image ou None si erreur critique
        """
        try:
            # Nettoyage du chemin source
            src_clean = src.replace('\\', '/')
            
            # Vérification de la longueur du nom de fichier
            filename = os.path.basename(src_clean)
            if len(filename) > 255:
                error_collector.add_error(
                    ErrorType.IMAGE_NAME_TOO_LONG,
                    str(file_path),
                    f"Nom d'image trop long: {filename[:50]}...",
                    f"Longueur: {len(filename)} caractères"
                )
                return {
                    'src': src,
                    'local_path': None,
                    'alt': alt_text,
                    'title': title_text,
                    'description': f"Image non traitable (nom trop long): {filename[:30]}...",
                    'found': False,
                    'error': 'filename_too_long'
                }
            
            # Gestion des URLs absolues vs relatives
            if src_clean.startswith(('http://', 'https://')):
                img_path = src_clean
                local_path = None
            else:
                # Recherche du fichier local
                img_path = file_path.parent / src_clean
                local_path = str(img_path)
                
                # Tentatives de localisation si fichier non trouvé
                if not img_path.exists():
                    # Recherche récursive dans l'arborescence
                    img_name = os.path.basename(src_clean)
                    found = False
                    try:
                        for root, dirs, files in os.walk(file_path.parent):
                            if img_name in files:
                                img_path = Path(root) / img_name
                                local_path = str(img_path)
                                found = True
                                break
                    except Exception as search_error:
                        error_collector.add_error(
                            ErrorType.IMAGE_NOT_FOUND,
                            str(file_path),
                            f"Erreur lors de la recherche de l'image: {img_name}",
                            str(search_error)
                        )
                        found = False
                    
                    if not found:
                        error_collector.add_error(
                            ErrorType.IMAGE_NOT_FOUND,
                            str(file_path),
                            f"Image non trouvée: {src}",
                            f"Recherchée dans {file_path.parent} et sous-dossiers"
                        )
                        return {
                            'src': src,
                            'local_path': None,
                            'alt': alt_text,
                            'title': title_text,
                            'description': f"Image non trouvée: {os.path.basename(src)}",
                            'found': False,
                            'error': 'not_found'
                        }
            
            # Contexte pour l'analyse
            context = f"Texte alternatif: {alt_text}, Titre: {title_text}" if alt_text or title_text else "Image technique de documentation"
            
            # Analyse du contenu visuel de l'image via Ollama
            try:
                if local_path and os.path.exists(local_path):
                    print(f"🖼️  Analyse de l'image: {os.path.basename(str(img_path))}")
                    description = self.image_analyzer.analyze_image(local_path, context)
                else:
                    description = f"Image externe ou non accessible: {alt_text or title_text or os.path.basename(src)}"
                    
            except Exception as analysis_error:
                error_collector.add_error(
                    ErrorType.IMAGE_ANALYSIS_FAILED,
                    str(file_path),
                    f"Échec de l'analyse de l'image: {os.path.basename(str(img_path))}",
                    str(analysis_error)
                )
                description = f"Image non analysée (erreur): {alt_text or title_text or os.path.basename(src)}"
            
            return {
                'src': src,
                'local_path': local_path,
                'alt': alt_text,
                'title': title_text,
                'description': description,
                'found': local_path and os.path.exists(local_path) if local_path else False
            }
            
        except Exception as general_error:
            error_collector.add_error(
                ErrorType.IMAGE_ANALYSIS_FAILED,
                str(file_path),
                f"Erreur générale lors du traitement de l'image: {src}",
                str(general_error)
            )
            return {
                'src': src,
                'local_path': None,
                'alt': alt_text,
                'title': title_text,
                'description': f"Image non traitable (erreur): {os.path.basename(src)}",
                'found': False,
                'error': 'processing_failed'
            }


class HTMLDocumentParser:
    """Parser spécialisé pour documents HTML techniques avec analyse d'images.
    
    Cette classe implémente un parseur robuste capable de traiter :
    - Documents HTML traditionnels avec en-têtes hiérarchiques
    - Applications Angular avec structure non-conventionnelle
    - Images intégrées avec analyse automatique du contenu
    - Tables de navigation et de contenu
    
    Le parser extrait le contenu sous forme de chunks sémantiquement
    cohérents, chacun accompagné de ses métadonnées et éléments multimédias.
    
    Attributes:
        base_directory: Répertoire racine pour la résolution des chemins relatifs
        image_analyzer: Instance de l'analyseur d'images Ollama
    """
    """Parser pour documents HTML techniques avec gestion d'erreurs robuste"""
    
    def __init__(self, base_directory: str, image_analyzer: OllamaImageAnalyzer = None):
        self.base_directory = Path(base_directory)
        self.image_analyzer = image_analyzer or OllamaImageAnalyzer()
        
    def parse_html_file(self, file_path: Path, error_collector: ErrorCollector = None) -> List[DocumentChunk]:
        """Parse un fichier HTML et extrait les chunks de contenu.
        
        Point d'entrée principal pour le traitement d'un document HTML.
        Gestion robuste des erreurs avec collecte centralisée.
        
        Args:
            file_path: Chemin vers le fichier HTML à traiter
            error_collector: Collecteur d'erreurs optionnel
            
        Returns:
            List[DocumentChunk]: Liste des chunks extraits du document
        """
        """Parse un fichier HTML et retourne les chunks avec gestion d'erreurs"""
        if error_collector is None:
            error_collector = ErrorCollector()
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extraction du titre
            title = self._extract_title(soup)
            
            # Extraction des chunks par sections avec gestion d'erreurs
            chunks = self._extract_sections(soup, file_path, title, error_collector)
            
            return chunks
            
        except Exception as e:
            error_collector.add_error(
                ErrorType.PARSING_ERROR,
                str(file_path),
                f"Erreur lors du parsing du fichier HTML",
                str(e)
            )
            return []
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extrait le titre du document"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
            
        return "Document technique"
    
    def _extract_sections(self, soup: BeautifulSoup, file_path: Path, doc_title: str, error_collector: ErrorCollector) -> List[DocumentChunk]:
        """Extrait les sections du document avec détection de structure.
        
        Méthode intelligente qui s'adapte à deux types de structures :
        1. HTML traditionnel avec en-têtes H1-H6
        2. Structure Angular avec tables et divs
        
        Inclut le nettoyage des scripts/styles et la gestion des commentaires.
        
        Args:
            soup: Objet BeautifulSoup du document parse
            file_path: Chemin du fichier source
            doc_title: Titre du document extrait
            error_collector: Collecteur pour les erreurs de traitement
            
        Returns:
            List[DocumentChunk]: Chunks extraits selon la structure détectée
        """
        """Extrait les sections du document avec gestion spéciale pour HTML Angular"""
        chunks = []
        
        try:
            # Suppression des scripts et styles
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            # Suppression des commentaires Angular
            for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
                comment.extract()
            
            # Recherche des sections traditionnelles avec validation
            sections = []
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                # Validation: s'assurer que c'est bien un en-tête valide
                if element.name and len(element.name) == 2 and element.name[0] == 'h' and element.name[1].isdigit():
                    sections.append(element)
                else:
                    error_collector.add_warning(f"Élément ignoré dans {file_path.name} (nom invalide): {element.name}")
            
            if sections:
                # Traitement section par section
                for i, section in enumerate(sections):
                    try:
                        chunk = self._process_section(section, file_path, doc_title, i, error_collector)
                        if chunk:
                            chunks.append(chunk)
                    except Exception as section_error:
                        error_collector.add_error(
                            ErrorType.PARSING_ERROR,
                            str(file_path),
                            f"Erreur lors du traitement de la section {i+1}: {section.get_text()[:50]}",
                            str(section_error)
                        )
            else:
                # Pas de sections H1-H6, traitement alternatif pour structure Angular
                try:
                    chunks.extend(self._extract_angular_sections(soup, file_path, doc_title, error_collector))
                except Exception as angular_error:
                    error_collector.add_error(
                        ErrorType.PARSING_ERROR,
                        str(file_path),
                        "Erreur lors du traitement de la structure Angular",
                        str(angular_error)
                    )
            
        except Exception as general_error:
            error_collector.add_error(
                ErrorType.PARSING_ERROR,
                str(file_path),
                "Erreur générale lors de l'extraction des sections",
                str(general_error)
            )
        
        return chunks
    
    def _extract_angular_sections(self, soup: BeautifulSoup, file_path: Path, doc_title: str, error_collector: ErrorCollector) -> List[DocumentChunk]:
        """Extrait le contenu de documents HTML Angular sans en-têtes traditionnels"""
        chunks = []
        
        try:
            main_content = soup.find('div', class_='content') or soup.find('div', class_='fr-view') or soup.find('body')
            
            if not main_content:
                return chunks
            
            # Recherche de tables avec liens (menu/navigation)
            tables = main_content.find_all('table')
            
            for table_idx, table in enumerate(tables):
                try:
                    links = table.find_all('a')
                    if links:
                        chunk = self._process_navigation_table(table, file_path, doc_title, table_idx, error_collector)
                        if chunk:
                            chunks.append(chunk)
                    else:
                        chunk = self._process_content_table(table, file_path, doc_title, table_idx, error_collector)
                        if chunk:
                            chunks.append(chunk)
                except Exception as table_error:
                    error_collector.add_error(
                        ErrorType.PARSING_ERROR,
                        str(file_path),
                        f"Erreur lors du traitement de la table {table_idx}",
                        str(table_error)
                    )
            
            # Extraction des paragraphes et contenus hors tables
            other_content = []
            for element in main_content.find_all(['p', 'div', 'span'], recursive=True):
                if not any(element in table or table in element.parents for table in tables):
                    text = element.get_text().strip()
                    if text and len(text) > 20:
                        other_content.append(text)
            
            if other_content:
                content_text = '\n'.join(other_content)
                chunk_id = hashlib.md5(f"{file_path}_other_content".encode()).hexdigest()
                chunk = DocumentChunk(
                    content=content_text,
                    source_file=str(file_path),
                    chunk_id=chunk_id,
                    title=doc_title,
                    heading="Contenu général"
                )
                chunks.append(chunk)
        
        except Exception as e:
            error_collector.add_error(
                ErrorType.PARSING_ERROR,
                str(file_path),
                "Erreur lors de l'extraction Angular",
                str(e)
            )
        
        return chunks
    
    def _process_navigation_table(self, table, file_path: Path, doc_title: str, table_idx: int, error_collector: ErrorCollector) -> Optional[DocumentChunk]:
        """Traite une table contenant des liens de navigation"""
        try:
            links_info = []
            link_texts = []
            
            for link in table.find_all('a'):
                href = link.get('href', '')
                text = link.get_text().strip()
                if text:
                    links_info.append({'text': text, 'href': href})
                    link_texts.append(text)
            
            if not link_texts:
                return None
            
            content = f"Navigation - {doc_title}\n\n"
            content += "Sections disponibles:\n"
            content += '\n'.join([f"- {link['text']}" for link in links_info])
            
            chunk_id = hashlib.md5(f"{file_path}_nav_table_{table_idx}".encode()).hexdigest()
            
            return DocumentChunk(
                content=content,
                source_file=str(file_path),
                chunk_id=chunk_id,
                title=doc_title,
                heading="Navigation",
                links=[link['href'] for link in links_info]
            )
        
        except Exception as e:
            error_collector.add_error(
                ErrorType.PARSING_ERROR,
                str(file_path),
                f"Erreur lors du traitement de la table de navigation {table_idx}",
                str(e)
            )
            return None
    
    def _process_content_table(self, table, file_path: Path, doc_title: str, table_idx: int, error_collector: ErrorCollector) -> Optional[DocumentChunk]:
        """Traite une table contenant du contenu (notamment des images)"""
        try:
            content_parts = []
            images = []
            
            text_content = table.get_text().strip()
            if text_content:
                content_parts.append(text_content)
            
            img_tags = table.find_all('img')
            for img in img_tags:
                img_info = self._process_image(img, file_path, error_collector)
                if img_info:
                    images.append(img_info)
                    content_parts.append(f"[Image: {img_info['description']}]")
            
            if not content_parts and not images:
                return None
            
            content = f"Contenu visuel - {doc_title}\n\n"
            if content_parts:
                content += '\n'.join(content_parts)
            
            chunk_id = hashlib.md5(f"{file_path}_content_table_{table_idx}".encode()).hexdigest()
            
            return DocumentChunk(
                content=content,
                source_file=str(file_path),
                chunk_id=chunk_id,
                title=doc_title,
                heading="Contenu multimédia",
                images=images
            )
        
        except Exception as e:
            error_collector.add_error(
                ErrorType.PARSING_ERROR,
                str(file_path),
                f"Erreur lors du traitement de la table de contenu {table_idx}",
                str(e)
            )
            return None
    
    def _process_section(self, section_element, file_path: Path, doc_title: str, section_index: int, error_collector: ErrorCollector) -> Optional[DocumentChunk]:
        """Traite une section spécifique avec validation améliorée"""
        try:
            heading = section_element.get_text().strip()
            
            # Validation améliorée du niveau de section
            if not (section_element.name and len(section_element.name) == 2 and 
                    section_element.name[0] == 'h' and section_element.name[1].isdigit()):
                error_collector.add_warning(f"Section ignorée dans {file_path.name} (nom invalide): {section_element.name}")
                return None
            
            content_elements = []
            current = section_element.next_sibling
            section_level = int(section_element.name[1])
            
            while current:
                if current.name and current.name.startswith('h'):
                    # Validation avant conversion
                    if len(current.name) == 2 and current.name[1].isdigit():
                        current_level = int(current.name[1])
                        if current_level <= section_level:
                            break
                    # Si nom invalide, on continue sans traiter comme en-tête
                content_elements.append(current)
                current = current.next_sibling
            
            text_content = []
            images = []
            links = []
            
            for element in content_elements:
                if hasattr(element, 'get_text'):
                    text = element.get_text().strip()
                    if text:
                        text_content.append(text)
                    
                    img_tags = element.find_all('img') if hasattr(element, 'find_all') else []
                    for img in img_tags:
                        img_info = self._process_image(img, file_path, error_collector)
                        if img_info:
                            images.append(img_info)
                    
                    link_tags = element.find_all('a') if hasattr(element, 'find_all') else []
                    for link in link_tags:
                        href = link.get('href')
                        if href and not href.startswith('#'):
                            links.append(href)
            
            content = f"{heading}\n\n" + "\n".join(text_content)
            
            if len(content.strip()) < 50:
                return None
            
            chunk_id = hashlib.md5(f"{file_path}_{section_index}_{heading}".encode()).hexdigest()
            
            return DocumentChunk(
                content=content,
                source_file=str(file_path),
                chunk_id=chunk_id,
                title=doc_title,
                heading=heading,
                images=images,
                links=links
            )
        
        except Exception as e:
            error_collector.add_error(
                ErrorType.PARSING_ERROR,
                str(file_path),
                f"Erreur lors du traitement de la section: {section_element.get_text()[:50]}",
                str(e)
            )
            return None
    
    def _process_image(self, img_tag, file_path: Path, error_collector: ErrorCollector) -> Optional[Dict]:
        """Traite une balise image avec localisation et analyse du contenu.
        
        Pipeline complet de traitement d'image :
        1. Validation et nettoyage du chemin source
        2. Localisation du fichier image (relatif/absolu)
        3. Recherche récursive si nécessaire
        4. Analyse du contenu visuel via Ollama
        5. Construction des métadonnées enrichies
        
        Gestion robuste des cas d'échec avec descriptions de fallback.
        
        Args:
            img_tag: Balise HTML <img> à traiter
            file_path: Chemin du document HTML parent
            error_collector: Collecteur pour les erreurs de traitement
            
        Returns:
            Optional[Dict]: Informations sur l'image ou None si échec critique
        """
        """Traite une image trouvée dans le document avec gestion d'erreurs robuste"""
        src = img_tag.get('src')
        if not src:
            return None
        
        try:
            src_clean = src.replace('\\', '/')
            
            # Vérification de la longueur du nom de fichier
            filename = os.path.basename(src_clean)
            if len(filename) > 255:  # Limite système de fichiers
                error_collector.add_error(
                    ErrorType.IMAGE_NAME_TOO_LONG,
                    str(file_path),
                    f"Nom d'image trop long (probablement base64 embed): {filename[:50]}...",
                    f"Longueur: {len(filename)} caractères"
                )
                # Retourner une description par défaut
                return {
                    'src': src,
                    'local_path': None,
                    'alt': img_tag.get('alt', ''),
                    'title': img_tag.get('title', ''),
                    'description': f"Image non traitable (nom trop long): {filename[:30]}...",
                    'found': False,
                    'error': 'filename_too_long'
                }
            
            # Gestion des URLs absolues
            if src_clean.startswith('http'):
                img_path = src_clean
                local_path = None
            else:
                # Recherche du fichier local
                img_path = file_path.parent / src_clean
                local_path = str(img_path)
                
                # Tentatives de localisation de l'image
                if not img_path.exists():
                    # Essai dans le dossier images/
                    if 'images/' in src_clean:
                        # Extraction du nom de fichier pour recherche alternative
                        img_name = os.path.basename(src_clean)
                        img_path = file_path.parent / img_name
                        local_path = str(img_path)
                    
                    # Recherche récursive dans l'arborescence si image toujours introuvable
                    if not img_path.exists():
                        img_name = os.path.basename(src_clean)
                        found = False
                        try:
                            # Parcours récursif de l'arborescence à partir du dossier parent
                            for root, dirs, files in os.walk(file_path.parent):
                                if img_name in files:
                                    img_path = Path(root) / img_name
                                    local_path = str(img_path)
                                    found = True
                                    break
                        except Exception as search_error:
                            error_collector.add_error(
                                ErrorType.IMAGE_NOT_FOUND,
                                str(file_path),
                                f"Erreur lors de la recherche de l'image: {img_name}",
                                str(search_error)
                            )
                            found = False
                        
                        if not found:
                            error_collector.add_error(
                                ErrorType.IMAGE_NOT_FOUND,
                                str(file_path),
                                f"Image non trouvée: {src}",
                                f"Recherchée dans {file_path.parent} et sous-dossiers"
                            )
                            # Continuer avec une description par défaut
                            return {
                                'src': src,
                                'local_path': None,
                                'alt': img_tag.get('alt', ''),
                                'title': img_tag.get('title', ''),
                                'description': f"Image non trouvée: {os.path.basename(src)}",
                                'found': False,
                                'error': 'not_found'
                            }
            
            # Extraction des métadonnées
            alt_text = img_tag.get('alt', '')
            title_text = img_tag.get('title', '')
            
            # Nettoyage des textes alternatifs qui sont en fait des noms de fichiers
            if alt_text and alt_text.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                alt_text = ""
            
            context = f"Texte alternatif: {alt_text}, Titre: {title_text}" if alt_text or title_text else "Image technique de documentation"
            
            # Analyse du contenu visuel de l'image via Ollama
            # Génération d'une description technique détaillée
            try:
                if local_path and os.path.exists(local_path):
                    print(f"🖼️  Analyse de l'image: {os.path.basename(str(img_path))}")
                    description = self.image_analyzer.analyze_image(local_path, context)
                else:
                    description = f"Image externe ou non accessible: {alt_text or title_text or os.path.basename(src)}"
                    
            except Exception as analysis_error:
                error_collector.add_error(
                    ErrorType.IMAGE_ANALYSIS_FAILED,
                    str(file_path),
                    f"Échec de l'analyse de l'image: {os.path.basename(str(img_path))}",
                    str(analysis_error)
                )
                description = f"Image non analysée (erreur): {alt_text or title_text or os.path.basename(src)}"
            
            return {
                'src': src,
                'local_path': local_path,
                'alt': alt_text,
                'title': title_text,
                'description': description,
                'found': local_path and os.path.exists(local_path) if local_path else False
            }
            
        except Exception as general_error:
            error_collector.add_error(
                ErrorType.IMAGE_ANALYSIS_FAILED,
                str(file_path),
                f"Erreur générale lors du traitement de l'image: {src}",
                str(general_error)
            )
            # Retourner un objet par défaut pour permettre au traitement de continuer
            return {
                'src': src,
                'local_path': None,
                'alt': img_tag.get('alt', ''),
                'title': img_tag.get('title', ''),
                'description': f"Image non traitable (erreur): {os.path.basename(src)}",
                'found': False,
                'error': 'processing_failed'
            }

class Qwen3Reranker:
    """Reranker utilisant le modèle Qwen3-Reranker-4B pour affiner les résultats de recherche.
    
    Cette classe implémente un système de reranking sémantique permettant
    d'améliorer la pertinence des résultats de recherche vectorielle.
    
    Le reranker évalue chaque paire (requête, document) et attribue
    un score de pertinence plus précis que la simple similarité cosinus.
    
    Optimisé pour différentes plateformes :
    - CUDA avec Flash Attention 2 pour les performances maximales
    - MPS (Mac) avec adaptations spécifiques
    - CPU en fallback
    
    Attributes:
        model_name: Nom du modèle Hugging Face utilisé
        use_flash_attention: Si Flash Attention 2 est activé
        tokenizer: Tokenizer pour le prétraitement du texte
        model: Modèle de classification pour le scoring
        device: Device PyTorch utilisé (cuda/mps/cpu)
    """
    """Reranker utilisant Qwen3-Reranker-4B"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-4B", use_flash_attention: bool = True):
        self.model_name = model_name
        self.use_flash_attention = use_flash_attention
        
        print(f"Chargement du reranker {model_name}...")
        
        # Détection de l'environnement
        self.is_mps = torch.backends.mps.is_available()
        self.is_cuda = torch.cuda.is_available()
        
        if self.is_mps:
            print("  - Détection de MPS (Mac), désactivation automatique de Flash Attention")
            self.use_flash_attention = False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model_kwargs = {
                "torch_dtype": torch.float16 if not self.is_mps else torch.float32,  # MPS ne supporte pas toujours float16
            }
            
            # Configuration spécifique selon la plateforme
            if self.is_cuda and use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("  - Flash Attention 2 activée (CUDA)")
                except Exception as flash_error:
                    print(f"  - Flash Attention 2 non disponible: {flash_error}")
                    print("  - Fallback vers attention standard")
                    self.use_flash_attention = False
            elif self.is_mps:
                model_kwargs["device_map"] = None  # device_map peut causer des problèmes avec MPS
                print("  - Configuration optimisée pour MPS")
            else:
                model_kwargs["device_map"] = "auto"
                print("  - Configuration CPU")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Configuration du device après chargement
            if self.is_mps:
                self.device = torch.device("mps")
                self.model = self.model.to(self.device)
            elif self.is_cuda:
                self.device = next(self.model.parameters()).device
            else:
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
            
            print(f"✅ Reranker chargé sur {self.device}")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du reranker: {e}")
            print("💡 Suggestions pour Mac/MPS:")
            print("   1. Essayez: --no-flash-attention")
            print("   2. Ou: --no-reranker pour désactiver complètement")
            print("   3. Vérifiez que PyTorch MPS est bien configuré")
            raise
    
    def _format_pair(self, query: str, document: str, instruction: str = None) -> str:
        """Formate une paire (requête, document) pour le reranker.
        
        Le format d'entrée influence la qualité du scoring. Une instruction
        spécialisée peut être ajoutée pour guider l'évaluation.
        
        Args:
            query: Requête utilisateur
            document: Document à évaluer
            instruction: Instruction optionnelle pour le contexte
            
        Returns:
            str: Paire formatée pour l'entrée du modèle
        """
        """Formate une paire query-document pour le reranker"""
        if instruction:
            return f"Instruction: {instruction}\nQuery: {query}\nDocument: {document}"
        return f"Query: {query}\nDocument: {document}"
    
    def rerank(self, query: str, documents: List[str], batch_size: int = 8, instruction: str = None) -> List[float]:
        """Reranke une liste de documents par rapport à une requête.
        
        Traitement par batch pour optimiser les performances et gérer
        la mémoire GPU. Utilise une instruction spécialisée pour le domaine
        de la construction et du béton.
        
        Args:
            query: Requête de recherche
            documents: Liste des documents à évaluer
            batch_size: Taille des batches pour le traitement
            instruction: Instruction personnalisée (optionnelle)
            
        Returns:
            List[float]: Scores de pertinence (0-1) pour chaque document
        """
        """Reranke une liste de documents par rapport à une requête"""
        if not documents:
            return []
        
        # Construction d'une instruction générique pour la documentation technique
        default_instruction = ("Évaluez la pertinence de ce document technique "
                             "par rapport à la requête en considérant : terminologie spécialisée, "
                             "concepts techniques, procédures, exemples de code et bonnes pratiques.")
        
        if instruction is None:
            instruction = default_instruction
        
        # Formatage des paires
        pairs = [self._format_pair(query, doc, instruction) for doc in documents]
        
        scores = []
        
        # Traitement par batch pour optimiser l'utilisation mémoire GPU/CPU
        # Chaque batch est traité indépendamment avec gestion d'erreurs
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            try:
                # Tokenisation des paires avec padding et troncature
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Déplacement vers le bon device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Inférence avec désactivation du calcul des gradients
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_scores = torch.nn.functional.sigmoid(outputs.logits)
                    
                    # Conversion des scores en numpy avec gestion des spécificités MPS
                    if self.is_mps:
                        # MPS peut nécessiter un passage explicite par CPU
                        batch_scores = batch_scores.cpu().numpy().flatten()
                    else:
                        batch_scores = batch_scores.cpu().numpy().flatten()
                    
                    scores.extend(batch_scores.tolist())
                    
            except Exception as batch_error:
                print(f"⚠️ Erreur lors du traitement du batch {i//batch_size + 1}: {batch_error}")
                # Fallback: scores neutres pour ce batch
                batch_size_actual = len(batch_pairs)
                scores.extend([0.5] * batch_size_actual)
                
        return scores
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du modèle"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'flash_attention': self.use_flash_attention,
            'parameters': sum(p.numel() for p in self.model.parameters()) / 1e9,  # En milliards
        }

class VectorIndexer:
    """Gestionnaire d'index vectoriel utilisant FAISS.
    
    FAISS offre d'excellentes performances pour la recherche vectorielle :
    - Ajout de vecteurs 10x plus rapide
    - Pas de blocage sur collection.add()
    - Meilleure gestion de la mémoire
    - Support GPU/CPU optimisé
    
    Attributes:
        faiss_indexer: Instance de l'indexeur FAISS
        embedding_model: Modèle SentenceTransformer pour les embeddings
        reranker: Instance du reranker Qwen3 (optionnel)
        tracker: Gestionnaire de tracking JSON
    """
    
    def __init__(self, db_path: str = "./faiss_index", collection_name: str = "tech_docs", 
                 use_flash_attention: bool = True, use_reranker: bool = True):
        self.db_path = db_path
        self.collection_name = collection_name
        self.use_flash_attention = use_flash_attention
        self.use_reranker = use_reranker
        self.vectors_added = 0
        
        # Initialisation du tracker JSON (même système)
        self.tracker = IndexingTracker(db_path)
        
        # Initialisation FAISS
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS n'est pas disponible. Installez avec: pip install faiss-cpu")
        # Utiliser directement db_path sans suffixe pour cohérence
        self.faiss_indexer = FAISSIndexer(db_path, dimension=2560)
        
        # Détection de l'environnement pour l'embedding model
        self.is_mps = torch.backends.mps.is_available()
        self.is_cuda = torch.cuda.is_available()
        
        print(f"🖥️  Détection GPU:")
        print(f"   - MPS disponible: {torch.backends.mps.is_available()}")
        print(f"   - CUDA disponible: {torch.cuda.is_available()}")
        print(f"   - Utilisation effective: {'MPS' if self.is_mps else ('CUDA' if self.is_cuda else 'AUCUN GPU')}")
        
        if not self.is_mps and not self.is_cuda:
            raise RuntimeError("Aucun GPU détecté. Ce script nécessite MPS (Mac) ou CUDA.")
            
        if self.is_mps:
            print(f"   - PyTorch MPS built: {torch.backends.mps.is_built()}")
            print(f"   - Device MPS: {torch.device('mps')}")
        
        if self.is_mps and use_flash_attention:
            print("  - MPS détecté: désactivation automatique de Flash Attention pour l'embedding model")
            use_flash_attention = False
        
        # Chargement du modèle d'embedding
        print("\n🔄 Initialisation du modèle d'embedding...")
        self._initialize_embedding_model(use_flash_attention)
        
        # Reranker Qwen3-Reranker-4B
        if use_reranker:
            try:
                effective_flash_attention = use_flash_attention and not self.is_mps
                self.reranker = Qwen3Reranker(use_flash_attention=effective_flash_attention)
            except Exception as e:
                print(f"❌ Erreur lors du chargement du reranker: {e}")
                print("🔄 Désactivation du reranking")
                print("💡 Pour Mac, essayez: python script.py --no-reranker")
                self.use_reranker = False
                self.reranker = None
        else:
            self.reranker = None
        
        print(f"\n✅ FAISS Vector Store initialisé avec {self.faiss_indexer.count()} vecteurs existants")

    def _initialize_embedding_model(self, use_flash_attention: bool):
        """Initialise le modèle d'embedding selon les capacités matérielles."""
        try:
            if use_flash_attention and self.is_cuda:
                print("  - Configuration avec Flash Attention 2 activée (CUDA)")
                try:
                    self.embedding_model = SentenceTransformer(
                        "Qwen/Qwen3-Embedding-4B",
                        model_kwargs={
                            "attn_implementation": "flash_attention_2", 
                            "device_map": "auto"
                        },
                        tokenizer_kwargs={"padding_side": "left"}
                    )
                except Exception as flash_error:
                    print(f"  - Flash Attention échoué: {flash_error}")
                    print("  - Fallback vers configuration standard")
                    self.embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
                    self.use_flash_attention = False
            else:
                print("  - Configuration standard (MPS/CPU ou Flash Attention désactivé)")
                model_kwargs = {}
                
                if self.is_mps:
                    model_kwargs = {
                        "torch_dtype": torch.float32,
                    }
                
                if model_kwargs:
                    self.embedding_model = SentenceTransformer(
                        "Qwen/Qwen3-Embedding-4B",
                        model_kwargs=model_kwargs,
                        tokenizer_kwargs={"padding_side": "left"}
                    )
                else:
                    self.embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
            
            # Déplacement explicite vers le bon device après initialisation
            if self.is_mps:
                print("  - Déplacement du modèle d'embedding vers MPS...")
                self.embedding_model = self.embedding_model.to(torch.device("mps"))
                print(f"  - Modèle maintenant sur device: {self.embedding_model.device}")
            elif self.is_cuda:
                print("  - Déplacement du modèle d'embedding vers CUDA...")
                self.embedding_model = self.embedding_model.to(torch.device("cuda"))
                print(f"  - Modèle maintenant sur device: {self.embedding_model.device}")
            else:
                print("  - Modèle d'embedding restera sur CPU")
                
            print("✅ Modèle Qwen3-Embedding-4B chargé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur avec Qwen3-Embedding-4B: {e}")
            print("🔄 Fallback vers le modèle multilingual MiniLM...")
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # Déplacement du modèle de fallback vers le bon device
            if self.is_mps:
                print("  - Déplacement du modèle de fallback vers MPS...")
                self.embedding_model = self.embedding_model.to(torch.device("mps"))
                print(f"  - Modèle de fallback sur device: {self.embedding_model.device}")
            elif self.is_cuda:
                print("  - Déplacement du modèle de fallback vers CUDA...")
                self.embedding_model = self.embedding_model.to(torch.device("cuda"))
                print(f"  - Modèle de fallback sur device: {self.embedding_model.device}")
                
            self.use_flash_attention = False

    def add_chunks(self, chunks: List[DocumentChunk], file_path: Path = None):
        """Ajoute des chunks à l'index FAISS avec performance optimisée."""
        if not chunks:
            return
        
        print(f"📝 Préparation de {len(chunks)} chunks pour indexation FAISS...")
        
        # Construction des documents et métadonnées (même code)
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Document enrichi avec descriptions d'images
            full_content = chunk.content
            
            # Ajout des descriptions d'images si présentes
            if chunk.images:
                img_descriptions = [img['description'] for img in chunk.images if img.get('description')]
                if img_descriptions:
                    full_content += "\n\nImages: " + " | ".join(img_descriptions)
            
            documents.append(full_content)
            
            # Métadonnées avec protection contre les chemins trop longs
            source_file = chunk.source_file
            if len(source_file) > 400:  # Limite conservative pour FAISS
                source_file = "..." + source_file[-397:]
                
            metadata = {
                'source_file': source_file,
                'title': chunk.title[:200] if chunk.title else "",
                'heading': chunk.heading[:200] if chunk.heading else "",
                'num_images': len(chunk.images),
                'num_links': len(chunk.links),
                'content_length': len(chunk.content),
                'chunk_content': chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
            }
            
            metadatas.append(metadata)
            ids.append(chunk.chunk_id)
        
        # Génération des embeddings (même code que avant)
        print(f"Génération des embeddings Qwen3 pour {len(documents)} chunks...")
        embeddings = self._generate_embeddings(documents)
        
        if not embeddings:
            print("❌ Aucun embedding généré")
            return
            
        # Ajout à FAISS avec protection contre les segfaults MPS
        print(f"⚡ Ajout à FAISS...")
        try:
            # Protection contre les conflits MPS/FAISS sur Mac
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🔄 Conversion des embeddings MPS vers CPU pour compatibilité FAISS...")

                # S'assurer que les embeddings sont en format CPU numpy
                if isinstance(embeddings, torch.Tensor):
                    embeddings_safe = embeddings.cpu().numpy().astype(np.float32)
                elif isinstance(embeddings, list) and len(embeddings) > 0:
                    if isinstance(embeddings[0], torch.Tensor):
                        embeddings_safe = [emb.cpu().numpy() if hasattr(emb, 'cpu') else emb for emb in embeddings]
                    else:
                        embeddings_safe = embeddings
                else:
                    embeddings_safe = embeddings

                # Vérification des valeurs invalides
                if isinstance(embeddings_safe, np.ndarray):
                    if not np.isfinite(embeddings_safe).all():
                        print("⚠️ Nettoyage des valeurs invalides dans les embeddings...")
                        embeddings_safe = np.nan_to_num(embeddings_safe, nan=0.0, posinf=1.0, neginf=-1.0)
                elif isinstance(embeddings_safe, list):
                    # Vérifier chaque embedding individuellement
                    for i, emb in enumerate(embeddings_safe):
                        if isinstance(emb, np.ndarray) and not np.isfinite(emb).all():
                            embeddings_safe[i] = np.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                embeddings_safe = embeddings

            # Ajout avec gestion d'erreur spécifique
            self.faiss_indexer.add_vectors(embeddings_safe, ids, metadatas)
            self.vectors_added += len(chunks)

            print(f"✅ {len(chunks)} chunks ajoutés à l'index FAISS")
            print(f"📊 Vecteurs ajoutés dans cette session: {self.vectors_added}")

            # Mise à jour du tracking JSON si un fichier est spécifié
            if file_path:
                total_images = sum(len(chunk.images) for chunk in chunks)
                self.tracker.mark_file_indexed(file_path, len(chunks), total_images)

        except Exception as e:
            print(f"❌ Erreur lors de l'ajout FAISS: {e}")
            print("💡 Solutions possibles:")
            print("   1. Redémarrez Python complètement")
            print("   2. Supprimez le dossier faiss_index et recommencez")
            print("   3. Utilisez --no-reranker si le problème persiste")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_embeddings(self, documents: List[str]) -> List[List[float]]:
        """Génère les embeddings pour un batch de textes."""
        # Vérification de la longueur moyenne des documents
        avg_length = sum(len(doc) for doc in documents) / len(documents)
        print(f"  - Longueur moyenne des chunks: {avg_length:.0f} caractères")
        
        if self.is_mps:
            if avg_length > 2000:
                batch_size = 1  # Ultra-conservateur pour longs documents sur MPS
                print(f"  - Mode MPS + chunks longs: traitement chunk par chunk")
            else:
                batch_size = 2  # Très conservateur pour MPS
                print(f"  - Mode MPS détecté: utilisation de batches de {batch_size} chunks")
        elif self.is_cuda:
            batch_size = 8 if avg_length < 1000 else 4
            print(f"  - Mode CUDA détecté: utilisation de batches de {batch_size} chunks")
        else:
            print(f"❌ Aucun GPU détecté (MPS/CUDA) - traitement impossible")
            raise RuntimeError("GPU requis pour le traitement des embeddings. Utilisez MPS (Mac) ou CUDA.")
        
        try:
            embeddings = []
            
            # Vérification du device utilisé pour les embeddings
            device_info = "GPU"
            if hasattr(self.embedding_model, 'device'):
                device_info = str(self.embedding_model.device)
            elif hasattr(self.embedding_model, '_modules') and self.embedding_model._modules:
                for module_name, module in self.embedding_model._modules.items():
                    if hasattr(module, 'parameters'):
                        try:
                            device_info = str(next(module.parameters()).device)
                            break
                        except StopIteration:
                            continue
            
            print(f"  - Device utilisé pour les embeddings: {device_info}")
            if "cpu" in device_info.lower():
                print("⚠️ ATTENTION: Le modèle semble être sur CPU au lieu de GPU")
                print("🚫 Cela peut indiquer un problème de configuration GPU")
            
            # Traitement par batch
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                
                print(f"  - Traitement du batch {batch_num}/{total_batches} ({len(batch_docs)} chunks)")
                
                try:
                    import time
                    print(f"    - Début génération embeddings pour batch {batch_num}...")
                    start_time = time.time()
                    
                    # Génération des embeddings pour ce batch avec monitoring
                    batch_embeddings = self.embedding_model.encode(batch_docs).tolist()
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"    - Embeddings générés pour batch {batch_num}: {len(batch_embeddings)} vecteurs en {duration:.1f}s")
                    embeddings.extend(batch_embeddings)
                    
                    # Sur MPS, forcer le nettoyage de la mémoire après chaque batch
                    if self.is_mps:
                        import torch
                        import gc
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        gc.collect()
                    
                except Exception as batch_error:
                    print(f"❌ Erreur critique lors du traitement du batch {batch_num}: {batch_error}")
                    print("🚫 Échec de génération embeddings - GPU requis pour tous les documents")
                    raise batch_error  # Pas de fallback CPU - échec immédiat
            
            print(f"✅ Embeddings générés par batch (dimension: {len(embeddings[0]) if embeddings else 'N/A'})")
            print(f"📊 Total embeddings: {len(embeddings)}/{len(documents)}")
            
            return embeddings
            
        except Exception as general_error:
            print(f"❌ Erreur générale lors de la génération d'embeddings: {general_error}")
            return []

    
    def get_stats(self) -> Dict:
        """Retourne les statistiques de l'index FAISS."""
        stats = self.faiss_indexer.get_stats()
        return {
            'initial_count': stats['total_vectors'] - stats['vectors_added_session'],
            'vectors_added_session': stats['vectors_added_session'],
            'total_count': stats['total_vectors'],
            'db_path': self.db_path,
            'backend': 'FAISS-HNSW',
            'dimension': stats['dimension']
        }
    
    def validate_mapping_integrity(self) -> Dict:
        """Valide l'intégrité du mapping entre l'index FAISS et les métadonnées.
        
        Returns:
            Dict: Rapport de validation avec statistiques et problèmes détectés
        """
        print("🔍 Validation de l'intégrité du mapping index ↔ métadonnées...")
        
        # Statistiques de base
        total_vectors = self.faiss_indexer.count()
        total_metadata = len(self.faiss_indexer.metadata)
        
        print(f"📊 Vecteurs FAISS: {total_vectors}")
        print(f"📊 Entrées métadonnées: {total_metadata}")
        
        problems = []
        valid_mappings = 0
        missing_metadata = []
        orphan_metadata = []
        empty_contents = []
        
        # Vérification du type de clés utilisées
        sample_keys = list(self.faiss_indexer.metadata.keys())[:3]
        uses_hash_keys = len(sample_keys) > 0 and all(len(key) == 32 and all(c in '0123456789abcdef' for c in key) for key in sample_keys)
        
        print(f"📋 Type de clés détecté: {'Hash MD5' if uses_hash_keys else 'Indices séquentiels'}")
        
        if uses_hash_keys:
            # Mode hash : vérifier que chaque vecteur a des métadonnées
            # Utiliser le fichier mappings.pkl pour faire la correspondance
            mappings_file = Path(self.db_path) / "mappings.pkl"
            if mappings_file.exists():
                import pickle
                with open(mappings_file, 'rb') as f:
                    mappings = pickle.load(f)
                    idx_to_id = mappings.get('idx_to_id', {})
                    
                print(f"📋 Mappings chargés: {len(idx_to_id)} entrées")
                
                for i in range(total_vectors):
                    if i in idx_to_id:
                        chunk_id = idx_to_id[i]
                        if chunk_id in self.faiss_indexer.metadata:
                            metadata = self.faiss_indexer.metadata[chunk_id]
                            if not metadata.get('chunk_content', '').strip():
                                empty_contents.append(i)
                            else:
                                valid_mappings += 1
                        else:
                            missing_metadata.append(i)
                    else:
                        missing_metadata.append(i)
            else:
                problems.append("Fichier mappings.pkl manquant pour les clés hash")
                # Marquer tous comme problématiques sans mappings
                missing_metadata.extend(range(total_vectors))
        else:
            # Mode séquentiel : vérification directe par index
            for i in range(total_vectors):
                metadata_key = str(i)
                if metadata_key in self.faiss_indexer.metadata:
                    metadata = self.faiss_indexer.metadata[metadata_key]
                    
                    # Vérification du contenu
                    if not metadata.get('chunk_content', '').strip():
                        empty_contents.append(i)
                    else:
                        valid_mappings += 1
                else:
                    missing_metadata.append(i)
        
        # Vérification des métadonnées orphelines (sans vecteur correspondant)
        if uses_hash_keys:
            # Pour les hash, vérifier avec mappings.pkl
            if mappings_file.exists():
                id_to_idx = mappings.get('id_to_idx', {})
                for key in self.faiss_indexer.metadata.keys():
                    if key not in id_to_idx:
                        orphan_metadata.append(key)
        else:
            # Pour les indices séquentiels
            for key in self.faiss_indexer.metadata.keys():
                try:
                    idx = int(key)
                    if idx >= total_vectors:
                        orphan_metadata.append(idx)
                except ValueError:
                    # Key non numérique - potentiel problème
                    problems.append(f"Clé métadonnée non numérique: {key}")
        
        # Construction du rapport
        is_healthy = (len(missing_metadata) == 0 and 
                     len(orphan_metadata) == 0 and 
                     len(empty_contents) == 0 and
                     len(problems) == 0)
        
        report = {
            'is_healthy': is_healthy,
            'total_vectors': total_vectors,
            'total_metadata': total_metadata,
            'valid_mappings': valid_mappings,
            'missing_metadata_count': len(missing_metadata),
            'orphan_metadata_count': len(orphan_metadata), 
            'empty_contents_count': len(empty_contents),
            'problems_count': len(problems),
            'missing_metadata_indices': missing_metadata[:10],  # Premiers 10
            'orphan_metadata_indices': orphan_metadata[:10],
            'empty_content_indices': empty_contents[:10],
            'problems': problems[:10]
        }
        
        # Affichage du rapport
        if is_healthy:
            print("✅ Mapping intègre: tous les vecteurs ont des métadonnées valides")
        else:
            print("❌ Problèmes de mapping détectés:")
            if missing_metadata:
                print(f"   • {len(missing_metadata)} vecteurs sans métadonnées: {missing_metadata[:5]}{'...' if len(missing_metadata) > 5 else ''}")
            if orphan_metadata:
                print(f"   • {len(orphan_metadata)} métadonnées orphelines: {orphan_metadata[:5]}{'...' if len(orphan_metadata) > 5 else ''}")
            if empty_contents:
                print(f"   • {len(empty_contents)} contenus vides: {empty_contents[:5]}{'...' if len(empty_contents) > 5 else ''}")
            if problems:
                print(f"   • {len(problems)} autres problèmes: {problems[:3]}")
        
        return report
    
    def file_needs_reindexing(self, file_path: Path) -> bool:
        """Vérifie si un fichier a besoin d'être réindexé (même logique)."""
        return self.tracker.needs_reindexing(file_path)


class UniversalDocumentParser:
    """Parser unifié pour documents HTML, Markdown et PDF avec analyse d'images.

    Cette classe combine les parsers HTML, Markdown et PDF pour traiter
    automatiquement différents types de documents techniques selon
    leur extension de fichier.

    Formats supportés :
    - HTML (.html, .htm) : Documents web, applications Angular
    - Markdown (.md, .markdown) : Documentation technique, README
    - PDF (.pdf) : Documents PDF avec chunking sémantique intelligent

    Attributes:
        base_directory: Répertoire racine pour la résolution des chemins
        html_parser: Instance du parser HTML
        markdown_parser: Instance du parser Markdown
        pdf_parser: Instance du parser PDF (si PyMuPDF disponible)
    """

    def __init__(self, base_directory: str):
        self.base_directory = Path(base_directory)

        # Créer une seule instance de l'analyseur d'images partagée
        shared_image_analyzer = OllamaImageAnalyzer()

        self.html_parser = HTMLDocumentParser(base_directory, shared_image_analyzer)
        self.markdown_parser = MarkdownDocumentParser(base_directory, shared_image_analyzer)

        # Initialisation du parser PDF
        try:
            from pdf_parser import PDFDocumentParser
            self.pdf_parser = PDFDocumentParser(
                chunk_size=1000,
                max_chunk_size=1500,
                min_chunk_size=300,
                chunk_overlap=100,
                use_semantic_chunking=True
            )
            self.pdf_support = True
            print("✅ Support PDF activé avec chunking sémantique intelligent")
        except ImportError as e:
            self.pdf_parser = None
            self.pdf_support = False
            print(f"⚠️ Support PDF désactivé. Installez PyMuPDF: pip install PyMuPDF")
    
    def parse_document(self, file_path: Path, error_collector: ErrorCollector = None) -> List[DocumentChunk]:
        """Parse un document selon son type automatiquement détecté.
        
        Args:
            file_path: Chemin vers le fichier à traiter
            error_collector: Collecteur d'erreurs optionnel
            
        Returns:
            List[DocumentChunk]: Chunks extraits du document
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.html', '.htm']:
            return self.html_parser.parse_html_file(file_path, error_collector)
        elif file_extension in ['.md', '.markdown']:
            return self.markdown_parser.parse_markdown_file(file_path, error_collector)
        elif file_extension == '.pdf' and self.pdf_support:
            return self._parse_pdf_document(file_path, error_collector)
        else:
            supported_formats = ".html, .htm, .md, .markdown"
            if self.pdf_support:
                supported_formats += ", .pdf"

            if error_collector:
                error_collector.add_error(
                    ErrorType.PARSING_ERROR,
                    str(file_path),
                    f"Format de fichier non supporté: {file_extension}",
                    f"Formats supportés: {supported_formats}"
                )
            return []

    def _parse_pdf_document(self, file_path: Path, error_collector: ErrorCollector = None) -> List[DocumentChunk]:
        """Parse un document PDF et convertit en DocumentChunks.

        Args:
            file_path: Chemin vers le fichier PDF
            error_collector: Collecteur d'erreurs optionnel

        Returns:
            List[DocumentChunk]: Chunks extraits du PDF
        """
        try:
            from pdf_parser import PDFChunk

            # Utilisation du parser PDF
            pdf_chunks = self.pdf_parser.parse_pdf_file(file_path)

            # Conversion des PDFChunks en DocumentChunks
            doc_chunks = []
            for pdf_chunk in pdf_chunks:
                doc_chunk = DocumentChunk(
                    content=pdf_chunk.content,
                    source_file=str(file_path),
                    chunk_id=pdf_chunk.metadata.get('chunk_id', ''),
                    title=pdf_chunk.section_title or file_path.stem,
                    heading=pdf_chunk.subsection_title or pdf_chunk.section_title or f"Page {pdf_chunk.page_numbers[0]+1 if pdf_chunk.page_numbers else 1}",
                    images=[],  # Les images PDF peuvent être extraites si nécessaire
                    links=[]    # Les liens PDF peuvent être extraits si nécessaire
                )
                doc_chunks.append(doc_chunk)

            return doc_chunks

        except Exception as e:
            if error_collector:
                error_collector.add_error(
                    ErrorType.PARSING_ERROR,
                    str(file_path),
                    f"Erreur lors du parsing PDF: {str(e)}",
                    "Le fichier PDF pourrait être corrompu ou protégé"
                )
            return []


class TechnicalDocIndexer:
    """Classe principale pour l'indexation de documentation technique universelle.
    
    Cette classe orchestre l'ensemble du processus d'indexation :
    - Analyse et parsing des documents HTML et Markdown
    - Extraction des images avec analyse par IA
    - Indexation vectorielle avec embeddings Qwen3
    - Recherche sémantique avec reranking
    
    Conçue pour traiter tout type de documentation technique :
    - Documentation de code et API
    - Guides d'installation et configuration
    - Tutoriels et exemples
    - Schémas et diagrammes architecturaux
    - Indexation incrémentale intelligente
    
    Attributes:
        docs_directory: Répertoire racine de la documentation
        parser: Instance du parser unifié HTML/Markdown
        indexer: Gestionnaire d'index vectoriel FAISS
        debug: Mode debug pour affichages détaillés
    """
    
    def __init__(self, docs_directory: str, db_path: str = "./faiss_index", debug: bool = False, 
                 use_flash_attention: bool = True, use_reranker: bool = True):
        self.docs_directory = Path(docs_directory)
        self.parser = UniversalDocumentParser(docs_directory)
        
        # Utiliser FAISS pour l'indexation vectorielle
        print("⚡ Utilisation FAISS (vectoriel haute performance)")
        self.indexer = VectorIndexer(db_path, use_flash_attention=use_flash_attention, use_reranker=use_reranker)
        
        self.debug = debug
        self.use_reranker = use_reranker
        
    def index_all_documents(self):
        """Indexe tous les documents HTML du répertoire (mode complet).
        
        Traitement exhaustif de tous les fichiers HTML trouvés dans
        l'arborescence du répertoire de documentation.
        
        Processus complet :
        1. Découverte récursive des fichiers HTML
        2. Parsing et extraction du contenu
        3. Analyse des images avec Ollama
        4. Génération des embeddings
        5. Stockage dans FAISS
        6. Statistiques détaillées
        
        Affichage en temps réel des erreurs et progrès.
        """
        print(f"🔍 Recherche des fichiers de documentation dans {self.docs_directory}")
        
        # Recherche de tous les formats supportés
        doc_files = (
            list(self.docs_directory.rglob("*.html")) +
            list(self.docs_directory.rglob("*.htm")) +
            list(self.docs_directory.rglob("*.md")) +
            list(self.docs_directory.rglob("*.markdown")) +
            list(self.docs_directory.rglob("*.pdf"))
        )
        
        if not doc_files:
            print("❌ Aucun fichier de documentation trouvé")
            return
        
        # Statistiques par type de fichier
        html_count = len([f for f in doc_files if f.suffix.lower() in ['.html', '.htm']])
        md_count = len([f for f in doc_files if f.suffix.lower() in ['.md', '.markdown']])
        pdf_count = len([f for f in doc_files if f.suffix.lower() == '.pdf'])
        
        print(f"📁 {len(doc_files)} fichier{'s' if len(doc_files) > 1 else ''} trouvé{'s' if len(doc_files) > 1 else ''} :")
        if html_count > 0:
            print(f"   • HTML: {html_count} fichier{'s' if html_count > 1 else ''}")
        if md_count > 0:
            print(f"   • Markdown: {md_count} fichier{'s' if md_count > 1 else ''}")
        if pdf_count > 0:
            print(f"   • PDF: {pdf_count} fichier{'s' if pdf_count > 1 else ''}")
        
        # Collecteur d'erreurs avec affichage immédiat
        error_collector = ErrorCollector(show_immediate=True)
        
        total_chunks = 0
        total_images_analyzed = 0
        successful_files = 0
        
        desc = f"Indexation {'du document' if len(doc_files) == 1 else 'des documents'}"
        for doc_file in tqdm(doc_files, desc=desc):
            file_status = "✅"
            try:
                print(f"\n📄 Traitement: {doc_file.name} ({doc_file.suffix})")
                
                # Utilisation du parser unifié avec collecteur d'erreurs
                chunks = self.parser.parse_document(doc_file, error_collector)
                
                if self.debug and chunks:
                    print(f"🔍 DEBUG - Chunks extraits de {doc_file.name}:")
                    for i, chunk in enumerate(chunks):
                        print(f"  Chunk {i+1}: {chunk.heading} ({len(chunk.content)} chars, {len(chunk.images)} images)")
                        if chunk.images:
                            for img in chunk.images:
                                print(f"    Image: {img['src']} -> {img['description'][:100]}...")
                
                if chunks:
                    file_images = sum(len(chunk.images) for chunk in chunks)
                    
                    try:
                        # Passage du file_path pour le tracking JSON
                        self.indexer.add_chunks(chunks, doc_file)
                        total_chunks += len(chunks)
                        total_images_analyzed += file_images
                        successful_files += 1
                        
                        print(f"   ✅ {len(chunks)} chunks extraits et indexés")
                        if file_images > 0:
                            print(f"   🖼️  {file_images} images traitées")
                            
                        current_session_vectors = self.indexer.vectors_added
                        print(f"   📊 Vecteurs session: {current_session_vectors}")
                            
                    except Exception as indexing_error:
                        error_collector.add_error(
                            ErrorType.EMBEDDING_ERROR,
                            str(html_file),
                            "Erreur lors de l'ajout à l'index",
                            str(indexing_error)
                        )
                        file_status = "⚠️"
                else:
                    print(f"   ⚠️  Aucun contenu exploitable")
                    file_status = "⚠️"
                    
            except Exception as file_error:
                error_collector.add_error(
                    ErrorType.PARSING_ERROR,
                    str(html_file),
                    "Erreur générale lors du traitement du fichier",
                    str(file_error)
                )
                file_status = "❌"
        
        # Statistiques finales
        stats = self.indexer.get_stats()
        print(f"\n🎉 Indexation terminée!")
        print(f"📊 Statistiques complètes:")
        print(f"   • Fichiers traités avec succès: {successful_files}/{len(doc_files)}")
        print(f"   • Chunks créés: {total_chunks}")
        print(f"   • Images traitées: {total_images_analyzed}")
        print(f"   • Vecteurs avant indexation: {stats['initial_count']}")
        print(f"   • Vecteurs ajoutés cette session: {stats['vectors_added_session']}")
        print(f"   • Total vecteurs en base: {stats['total_count']}")
        print(f"   • Collection: FAISS-HNSW")
        print(f"   • Chemin base: {stats['db_path']}")
        print(f"   • Backend: {stats['backend']}")
        print(f"   • Dimension: {stats['dimension']}")
        
        if total_images_analyzed > 0:
            print(f"\n🖼️  Analyse d'images réussie avec Ollama (qwen2.5vl:7b)")
        
        if self.use_reranker:
            print(f"\n🎯 Reranker configuré: Qwen3-Reranker")
        
        # Validation de l'intégrité du mapping
        print()
        mapping_report = self.indexer.validate_mapping_integrity()
        
        # Affichage du résumé compact à la fin
        error_collector.print_summary()

    def index_all_documents_incremental(self):
        """Indexation incrémentale avec tracking JSON optimisé.
        
        Version intelligente qui ne traite que les fichiers modifiés
        ou nouveaux, en s'appuyant sur le système de tracking JSON.
        
        Optimisations :
        - Vérification rapide via timestamps (mtime)
        - Suppression automatique des chunks obsolètes
        - Nettoyage des fichiers supprimés du tracking
        - Statistiques avant/après pour le suivi
        
        Idéal pour les mises à jour régulières de documentation.
        """
        """Version simplifiée de l'indexation incrémentale avec tracking JSON"""
        print(f"🔍 Recherche des fichiers HTML dans {self.docs_directory}")
        
        html_files = list(self.docs_directory.rglob("*.html")) + list(self.docs_directory.rglob("*.htm"))
        
        if not html_files:
            print("❌ Aucun fichier HTML trouvé")
            return
        
        print(f"📁 {len(html_files)} fichiers HTML trouvés")
        
        # Nettoyage du tracking (supprime les fichiers qui n'existent plus)
        deleted_count = self.indexer.cleanup_tracking(html_files)
        
        # Analyse rapide de l'état d'indexation via le tracker JSON
        # Permet d'éviter le scan complet de l'index vectoriel
        print("🔍 Vérification de l'état de l'indexation...")
        
        files_to_process = []
        files_to_reindex = []
        files_already_indexed = []
        
        # Classification des fichiers selon leur état d'indexation
        for html_file in html_files:
            if self.indexer.file_needs_reindexing(html_file):
                if self.indexer.file_already_indexed(html_file):
                    files_to_reindex.append(html_file)
                else:
                    files_to_process.append(html_file)
            else:
                files_already_indexed.append(html_file)
        
        # Affichage des statistiques de tracking
        tracking_stats = self.indexer.get_tracking_stats()
        
        print(f"📊 État de l'indexation:")
        print(f"   • Nouveaux fichiers à traiter: {len(files_to_process)}")
        print(f"   • Fichiers modifiés à réindexer: {len(files_to_reindex)}")
        print(f"   • Fichiers déjà à jour: {len(files_already_indexed)}")
        if deleted_count > 0:
            print(f"   • Fichiers supprimés nettoyés: {deleted_count}")
        
        if tracking_stats['total_files'] > 0:
            print(f"📋 Historique d'indexation:")
            print(f"   • Total fichiers trackés: {tracking_stats['total_files']}")
            print(f"   • Total chunks: {tracking_stats['total_chunks']}")
            print(f"   • Total images: {tracking_stats['total_images']}")
            if tracking_stats['newest_index']:
                newest = tracking_stats['newest_index'][:19].replace('T', ' ')
                print(f"   • Dernière indexation: {newest}")
        
        # Vérification si une indexation est nécessaire
        if not files_to_process and not files_to_reindex:
            print("✅ Tous les fichiers sont déjà indexés et à jour !")
            stats = self.indexer.get_stats()
            print(f"📊 Total vecteurs en base: {stats['total_count']}")
            return
        
        # Collecteur d'erreurs avec affichage immédiat
        error_collector = ErrorCollector(show_immediate=True)
        
        total_chunks = 0
        total_images_analyzed = 0
        successful_files = 0
        
        all_files_to_process = files_to_process + files_to_reindex
        
        for html_file in tqdm(all_files_to_process, desc="Indexation incrémentale"):
            file_status = "✅"
            is_reindexing = html_file in files_to_reindex
            
            try:
                print(f"\n📄 {'Réindexation' if is_reindexing else 'Traitement'}: {html_file.name}")
                
                # Suppression des anciens chunks si réindexation
                if is_reindexing:
                    self.indexer.remove_file_chunks(html_file)
                
                # Traitement du fichier
                chunks = self.parser.parse_html_file(html_file, error_collector)
                
                if self.debug and chunks:
                    print(f"🔍 DEBUG - Chunks extraits de {html_file.name}:")
                    for i, chunk in enumerate(chunks):
                        print(f"  Chunk {i+1}: {chunk.heading} ({len(chunk.content)} chars, {len(chunk.images)} images)")
                
                if chunks:
                    file_images = sum(len(chunk.images) for chunk in chunks)
                    
                    try:
                        # Passage du file_path pour le tracking JSON
                        self.indexer.add_chunks(chunks, html_file)
                        total_chunks += len(chunks)
                        total_images_analyzed += file_images
                        successful_files += 1
                        
                        print(f"   ✅ {len(chunks)} chunks {'réindexés' if is_reindexing else 'indexés'}")
                        if file_images > 0:
                            print(f"   🖼️  {file_images} images traitées")
                            
                    except Exception as indexing_error:
                        error_collector.add_error(
                            ErrorType.EMBEDDING_ERROR,
                            str(html_file),
                            "Erreur lors de l'ajout à l'index",
                            str(indexing_error)
                        )
                        file_status = "⚠️"
                else:
                    print(f"   ⚠️  Aucun contenu exploitable")
                    file_status = "⚠️"
                    
            except Exception as file_error:
                error_collector.add_error(
                    ErrorType.PARSING_ERROR,
                    str(html_file),
                    "Erreur générale lors du traitement du fichier",
                    str(file_error)
                )
                file_status = "❌"
        
        # Statistiques finales
        stats = self.indexer.get_stats()
        print(f"\n🎉 Indexation incrémentale terminée!")
        print(f"📊 Statistiques de cette session:")
        print(f"   • Fichiers traités: {successful_files}/{len(all_files_to_process)}")
        print(f"   • Fichiers déjà à jour ignorés: {len(files_already_indexed)}")
        print(f"   • Chunks créés: {total_chunks}")
        print(f"   • Images traitées: {total_images_analyzed}")
        print(f"   • Total vecteurs en base: {stats['total_count']}")
        
        # Affichage du résumé des erreurs
        error_collector.print_summary()
    
    def index_single_file(self, file_path: str):
        """Indexe un fichier unique avec debug détaillé.
        
        Mode debug pour analyser en détail le traitement d'un fichier
        spécifique. Affiche la structure HTML, les chunks générés,
        et le traitement des images.
        
        Utile pour :
        - Déboguer les problèmes de parsing
        - Comprendre la segmentation en chunks
        - Vérifier l'analyse des images
        - Tester les modifications du parser
        
        Args:
            file_path: Chemin vers le fichier HTML à traiter
        """
        """Index un seul fichier avec debug détaillé et gestion d'erreurs"""
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ Fichier non trouvé: {file_path}")
            return
            
        print(f"📄 Indexation de: {file_path}")
        
        # Collecteur d'erreurs pour ce fichier avec affichage immédiat
        error_collector = ErrorCollector(show_immediate=True)
        
        chunks = self.parser.parse_document(file_path, error_collector)
        
        if self.debug:
            print(f"🔍 DEBUG - Structure HTML analysée:")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
            
            print(f"  - Titre: {soup.find('title').get_text() if soup.find('title') else 'Non trouvé'}")
            print(f"  - En-têtes H1-H6: {len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))}")
            print(f"  - Tables: {len(soup.find_all('table'))}")
            print(f"  - Images: {len(soup.find_all('img'))}")
            print(f"  - Liens: {len(soup.find_all('a'))}")
        
        if chunks:
            if self.debug:
                print(f"🔍 DEBUG - Chunks créés:")
                for i, chunk in enumerate(chunks):
                    print(f"  Chunk {i+1}: '{chunk.heading}'")
                    print(f"    Contenu: {len(chunk.content)} caractères")
                    print(f"    Images: {len(chunk.images)}")
                    print(f"    Liens: {len(chunk.links)}")
                    if chunk.images:
                        for img in chunk.images:
                            status = "✅" if img.get('found', False) else "❌"
                            print(f"      {status} {img['src']} -> {img['description'][:80]}...")
                    print(f"    Aperçu: {chunk.content[:200]}...")
                    print()
            
            try:
                # Passage du file_path pour le tracking JSON
                self.indexer.add_chunks(chunks, file_path)
                print(f"✅ {len(chunks)} chunks ajoutés")
                
                total_images = sum(len(chunk.images) for chunk in chunks)
                if total_images > 0:
                    print(f"🖼️  {total_images} images traitées")
                
                stats = self.indexer.get_stats()
                print(f"📊 Vecteurs ajoutés: {stats['vectors_added_session']}, Total: {stats['total_count']}")
                
            except Exception as indexing_error:
                error_collector.add_error(
                    ErrorType.EMBEDDING_ERROR,
                    str(file_path),
                    "Erreur lors de l'ajout à l'index",
                    str(indexing_error)
                )
        else:
            print("⚠️ Aucun contenu exploitable")
        
        # Affichage des erreurs pour ce fichier
        error_collector.print_summary()

def main():
    """Point d'entrée principal du script avec gestion des arguments.
    
    Interface en ligne de commande complète avec options pour :
    - Indexation complète ou incrémentale
    - Test de recherche sémantique
    - Traitement de fichier unique
    - Configuration des modèles et plateformes
    - Mode debug détaillé
    
    Détection automatique de l'environnement (Mac/MPS, CUDA, CPU)
    avec suggestions d'optimisation.
    """
    """Fonction principale avec système de tracking JSON simplifié"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Indexation universelle de documentation technique avec Qwen3")
    parser.add_argument("docs_dir", help="Répertoire contenant les fichiers de documentation (HTML/Markdown)")
    parser.add_argument("--db-path", default="./faiss_index", help="Chemin de l'index FAISS")
    parser.add_argument("--single-file", help="Index un seul fichier au lieu du répertoire entier")
    parser.add_argument("--debug", action="store_true", help="Active le mode debug avec informations détaillées")
    parser.add_argument("--no-flash-attention", action="store_true", help="Désactive Flash Attention 2 (recommandé pour Mac)")
    parser.add_argument("--no-reranker", action="store_true", help="Désactive le reranking (utile si problèmes GPU)")
    
    # Options nouvelles
    parser.add_argument("--incremental", action="store_true", help="Mode incrémental : ne traite que les nouveaux fichiers ou modifiés")
    
    args = parser.parse_args()
    
    use_flash_attention = not args.no_flash_attention
    use_reranker = not args.no_reranker
    
    # Détection automatique de l'environnement et conseils d'optimisation
    is_mac = torch.backends.mps.is_available()
    is_cuda = torch.cuda.is_available()
    
    if is_mac:
        print("🍎 Mac avec MPS détecté")
        if use_flash_attention:
            print("💡 Conseil: Sur Mac, utilisez --no-flash-attention pour éviter les erreurs")
        print("💡 Si problèmes avec le reranker, utilisez --no-reranker")
        print("💡 FAISS est utilisé pour l'indexation vectorielle haute performance")
        print()
    elif is_cuda:
        print("🚀 GPU CUDA détecté - Configuration haute performance")
    else:
        print("❌ Aucun GPU détecté (MPS/CUDA)")
        print("🚫 Ce script nécessite un GPU pour fonctionner")
        print("   - Sur Mac: Assurez-vous que MPS est activé")
        print("   - Sur Linux/Windows: Installez CUDA et PyTorch avec support GPU")
        return
    
    indexer = TechnicalDocIndexer(
        args.docs_dir, 
        args.db_path, 
        debug=args.debug, 
        use_flash_attention=use_flash_attention,
        use_reranker=use_reranker
    )
    
    if args.single_file:
        indexer.index_single_file(args.single_file)
    else:
        if args.incremental:
            indexer.index_all_documents_incremental()
        else:
            indexer.index_all_documents()

if __name__ == "__main__":
    main()