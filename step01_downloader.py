#!/usr/bin/env python3
"""
Step01 - Téléchargement automatique de documentation web
Documentation Source Downloader with Selenium + ChromeDriver

Ce script télécharge automatiquement de la documentation technique depuis des sites web
et la convertit en fichiers Markdown pour indexation RAG ultérieure.

Fonctionnalités:
- Navigation automatique avec Selenium + Chrome
- Contournement Cloudflare avec intervention manuelle
- Conversion HTML → Markdown
- Découverte récursive de liens
- Processing multi-threadé
- Sauvegarde incrémentale (reprise possible)
- Structure pour intégration RAG

Usage:
    python step01_downloader.py [options]

Options:
    --url URL            URL de départ (défaut: documentation configurée)
    --output DIR         Répertoire de sortie (défaut: ./downloaded_docs)
    --workers N          Nombre de threads (défaut: 3)
    --resume             Reprendre téléchargement interrompu

Workflow suivant:
    step01_downloader.py → step02_indexer.py → step03_upload_embeddings.py → step04_chatbot.py
"""

import os
import time
import random
import json
import urllib.parse
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Set, List, Optional, Dict
from pathlib import Path

try:
    import chromedriver_autoinstaller
    import html2text
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.common.exceptions import TimeoutException
    from bs4 import BeautifulSoup
    from readability import Document
except ImportError as e:
    print(f"❌ Dépendance manquante: {e}")
    print("📦 Installation requise:")
    print("   pip install selenium beautifulsoup4 html2text chromedriver-autoinstaller readability-lxml")
    exit(1)

# Configuration globale
DEFAULT_OUTPUT_DIR = "./downloaded_docs"
DEFAULT_MAX_WORKERS = 3
SITEMAP_FILE = "sitemap.json"
BACKLOG_FILE = "backlog.json"


class PageStatus(Enum):
    """Status de traitement d'une page."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class PageInfo:
    """Information sur une page à traiter."""
    url: str
    status: PageStatus = PageStatus.PENDING
    discovered_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class PageBacklog:
    """Gestionnaire thread-safe du backlog de pages."""

    def __init__(self, output_dir: str, backlog_file: str = BACKLOG_FILE):
        self.output_dir = Path(output_dir)
        self.backlog_file = self.output_dir / backlog_file
        self.pages: Dict[str, PageInfo] = {}
        self.lock = threading.RLock()
        self.load_backlog()

    def load_backlog(self) -> None:
        """Charge le backlog existant depuis le disque."""
        with self.lock:
            if self.backlog_file.exists():
                try:
                    with open(self.backlog_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    for url, page_data in data.items():
                        page_info = PageInfo(
                            url=url,
                            status=PageStatus(page_data.get("status", "pending")),
                            discovered_at=datetime.fromisoformat(page_data.get("discovered_at", datetime.now().isoformat())),
                            started_at=datetime.fromisoformat(page_data["started_at"]) if page_data.get("started_at") else None,
                            completed_at=datetime.fromisoformat(page_data["completed_at"]) if page_data.get("completed_at") else None,
                            worker_id=page_data.get("worker_id"),
                            error_message=page_data.get("error_message"),
                            retry_count=page_data.get("retry_count", 0)
                        )
                        self.pages[url] = page_info

                    print(f"✅ Backlog chargé avec {len(self.pages)} pages")
                    self._print_status_summary()

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"⚠️ Erreur lors du chargement du backlog: {e} - nouveau backlog créé")
                    self.pages = {}

    def save_backlog(self) -> None:
        """Sauvegarde le backlog actuel sur disque."""
        with self.lock:
            self.output_dir.mkdir(exist_ok=True)

            data = {}
            for url, page_info in self.pages.items():
                data[url] = {
                    "status": page_info.status.value,
                    "discovered_at": page_info.discovered_at.isoformat(),
                    "started_at": page_info.started_at.isoformat() if page_info.started_at else None,
                    "completed_at": page_info.completed_at.isoformat() if page_info.completed_at else None,
                    "worker_id": page_info.worker_id,
                    "error_message": page_info.error_message,
                    "retry_count": page_info.retry_count
                }

            with open(self.backlog_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def add_page(self, url: str) -> bool:
        """Ajoute une nouvelle page au backlog si pas déjà présente."""
        with self.lock:
            if url not in self.pages:
                self.pages[url] = PageInfo(url=url)
                return True
            return False

    def get_next_pending_page(self, worker_id: str) -> Optional[PageInfo]:
        """Récupère la prochaine page en attente et la marque en cours."""
        with self.lock:
            for page_info in self.pages.values():
                if page_info.status == PageStatus.PENDING:
                    page_info.status = PageStatus.IN_PROGRESS
                    page_info.started_at = datetime.now()
                    page_info.worker_id = worker_id
                    return page_info
            return None

    def mark_completed(self, url: str) -> None:
        """Marque une page comme terminée."""
        with self.lock:
            if url in self.pages:
                self.pages[url].status = PageStatus.COMPLETED
                self.pages[url].completed_at = datetime.now()

    def mark_error(self, url: str, error_message: str) -> None:
        """Marque une page en erreur avec message."""
        with self.lock:
            if url in self.pages:
                page_info = self.pages[url]
                page_info.status = PageStatus.ERROR
                page_info.error_message = error_message
                page_info.retry_count += 1

    def get_stats(self) -> Dict[str, int]:
        """Retourne les statistiques de traitement."""
        with self.lock:
            stats = {status.value: 0 for status in PageStatus}
            for page_info in self.pages.values():
                stats[page_info.status.value] += 1
            stats["total"] = len(self.pages)
            return stats

    def _print_status_summary(self) -> None:
        """Affiche un résumé du statut actuel."""
        stats = self.get_stats()
        print(f"   📊 Statut: {stats['pending']} en attente, {stats['in_progress']} en cours, {stats['completed']} terminées, {stats['error']} erreurs")


class WebDocumentDownloader:
    """Téléchargeur de documentation web avec Selenium."""

    def __init__(self, output_dir: str, base_url: str, max_workers: int = 3):
        self.output_dir = Path(output_dir)
        self.base_url = base_url
        self.max_workers = max_workers
        self.backlog = PageBacklog(output_dir)

        # Installer ChromeDriver automatiquement
        try:
            chromedriver_autoinstaller.install()
        except Exception as e:
            print(f"⚠️ Erreur installation ChromeDriver: {e}")

    def create_driver(self) -> webdriver.Chrome:
        """Crée une instance ChromeDriver configurée."""
        options = Options()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--start-maximized")
        return webdriver.Chrome(options=options)

    def wait_for_page(self, driver: webdriver.Chrome, timeout: int = 30) -> None:
        """Attend le chargement complet de la page."""
        try:
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(random.uniform(1, 3))
        except TimeoutException:
            print("⚠️ Timeout d'attente de chargement")

    def detect_cloudflare_challenge(self, html: str) -> str:
        """Détecte le type de challenge Cloudflare."""
        loading_indicators = [
            "We are checking your browser",
            "Enable JavaScript and cookies",
            "Vérification réussie",
            "Checking if the site connection is secure",
            "This process is automatic"
        ]

        captcha_indicators = [
            "cf-challenge-form",
            "cf-turnstile",
            "Please complete the security check",
            "DDoS protection by Cloudflare",
            "cf-browser-verification",
            "cf-challenge",
            "Ray ID:",
        ]

        for indicator in loading_indicators:
            if indicator.lower() in html.lower():
                return "loading"

        for indicator in captcha_indicators:
            if indicator.lower() in html.lower():
                return "captcha"

        return ""

    def alert_user_cloudflare(self, driver: webdriver.Chrome, issue_type: str) -> None:
        """Alerte l'utilisateur pour intervention Cloudflare."""
        print("\n" + "="*60)
        print(f"🚨 ALERTE CLOUDFLARE - {issue_type.upper()}")
        print("="*60)
        print("⚠️  Intervention manuelle nécessaire !")
        print("📋 Actions à effectuer :")
        print("   1. Regardez la fenêtre Chrome ouverte")
        print("   2. Résolvez le captcha/défi Cloudflare")
        print("   3. Attendez le chargement complet")
        print("   4. Appuyez sur ENTRÉE pour continuer")
        print(f"\n🖥️  URL: {driver.current_url}")
        print("="*60)

        # Activer Chrome sur macOS
        try:
            import subprocess
            subprocess.run(["osascript", "-e", 'tell application "Google Chrome" to activate'],
                          capture_output=True, timeout=5)
        except:
            pass

        input("\n👆 Appuyez sur ENTRÉE après résolution du défi...")
        print("✅ Reprise du téléchargement...")

    def wait_for_real_content(self, driver: webdriver.Chrome, timeout: int = 300) -> str:
        """Attend le contenu réel après contournement Cloudflare."""
        start_time = time.time()
        user_alerted = False

        while time.time() - start_time < timeout:
            html = driver.page_source
            challenge_type = self.detect_cloudflare_challenge(html)

            if challenge_type == "captcha":
                if not user_alerted:
                    self.alert_user_cloudflare(driver, "captcha")
                    user_alerted = True
                    start_time = time.time()
                else:
                    time.sleep(2)
            elif challenge_type == "loading":
                print("🔄 Vérification Cloudflare automatique...")
                time.sleep(random.uniform(2, 5))
            else:
                if user_alerted:
                    print("✅ Challenge Cloudflare résolu")
                return html

        print("⚠️ Timeout Cloudflare - sauvegarde contenu actuel")
        return html

    def extract_content(self, html_content: str, url: str = "") -> str:
        """Extrait le contenu principal avec Readability (algorithme Mozilla)."""
        try:
            # Utiliser Readability pour extraire le contenu principal
            doc = Document(html_content)

            # Extraire le titre et le contenu principal
            title = doc.title()
            content = doc.summary()

            # Vérifier si le contenu est suffisant
            content_length = len(content) if content else 0

            if content_length < 50:
                print(f"⚠️ Readability: contenu trop court ({content_length} chars), fallback BeautifulSoup")
                return self._fallback_extract(html_content)
            elif title and content:
                # Combiner titre et contenu
                clean_html = f"<h1>{title}</h1>\n{content}"
                print(f"📄 Contenu extrait avec Readability: {title[:50]}...")
                return clean_html
            elif content:
                print("📄 Contenu extrait avec Readability (sans titre)")
                return content
            else:
                print("⚠️ Readability n'a pas trouvé de contenu, fallback BeautifulSoup")
                return self._fallback_extract(html_content)

        except Exception as e:
            print(f"❌ Erreur Readability: {e}, fallback BeautifulSoup")
            return self._fallback_extract(html_content)

    def _fallback_extract(self, html_content: str) -> str:
        """Extraction de fallback avec BeautifulSoup si Readability échoue."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Suppression des éléments non-documentaires
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()

            # Recherche du contenu principal avec plusieurs sélecteurs
            selectors = [
                "main", "article", ".content", ".post-content",
                ".entry-content", "#content", ".main-content",
                "div[role='main']", ".page-content"
            ]

            main_content = None
            for selector in selectors:
                main_content = soup.select_one(selector)
                if main_content and len(main_content.get_text().strip()) > 50:
                    print(f"📄 Fallback extraction via: {selector}")
                    return str(main_content)

            # Dernier recours: chercher les divs avec beaucoup de texte
            all_divs = soup.find_all("div")
            if all_divs:
                text_divs = [(div, len(div.get_text().strip())) for div in all_divs]
                text_divs.sort(key=lambda x: x[1], reverse=True)

                if text_divs and text_divs[0][1] > 200:
                    print(f"📄 Extraction agressive: div avec {text_divs[0][1]} caractères")
                    return str(text_divs[0][0])

            # Fallback final: body nettoyé
            body = soup.find("body")
            if body:
                print("📄 Fallback extraction: body")
                return str(body)

            return html_content

        except Exception as e:
            print(f"❌ Erreur fallback: {e}")
            return html_content

    def convert_to_markdown(self, html_content: str, url: str) -> str:
        """Convertit HTML en Markdown."""
        filtered_html = self.extract_content(html_content, url)

        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0
        h.unicode_snob = True
        h.bypass_tables = False

        md_content = h.handle(filtered_html)
        return md_content

    def save_page(self, url: str, html_content: str) -> None:
        """Sauvegarde une page en Markdown."""
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path.lstrip("/")

        if path.endswith("/") or not os.path.splitext(path)[1]:
            path = os.path.join(path, "index.md")
        else:
            path = os.path.splitext(path)[0] + ".md"

        file_path = self.output_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        md_content = self.convert_to_markdown(html_content, url)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"💾 Sauvegardé: {file_path}")

    def discover_links(self, html_content: str, base_url: str) -> List[str]:
        """Découvre les liens dans le contenu HTML."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            discovered_links = []

            for link in soup.find_all("a", href=True):
                href = urllib.parse.urljoin(base_url, link["href"])

                if (href.startswith(self.base_url) and
                    "#" not in href and
                    href != base_url):
                    discovered_links.append(href)

            return list(set(discovered_links))
        except Exception as e:
            print(f"❌ Erreur découverte liens: {e}")
            return []

    def process_page(self, driver: webdriver.Chrome, url: str, worker_id: str) -> bool:
        """Traite une page individuelle."""
        try:
            print(f"[Worker-{worker_id}] 📄 Traitement: {url}")

            driver.get(url)
            self.wait_for_page(driver)
            html = self.wait_for_real_content(driver)

            self.save_page(url, html)

            # Découverte de nouveaux liens
            new_links = self.discover_links(html, url)
            added_count = 0
            for link in new_links:
                if self.backlog.add_page(link):
                    added_count += 1

            if added_count > 0:
                print(f"[Worker-{worker_id}] 🔗 {added_count} nouveaux liens découverts")

            return True

        except Exception as e:
            print(f"[Worker-{worker_id}] ❌ Erreur: {e}")
            return False


class DownloadWorker:
    """Worker de téléchargement individuel."""

    def __init__(self, worker_id: str, downloader: WebDocumentDownloader):
        self.worker_id = worker_id
        self.downloader = downloader
        self.driver: Optional[webdriver.Chrome] = None
        self.processed_count = 0

    def start(self) -> None:
        """Démarre le worker."""
        try:
            print(f"[Worker-{self.worker_id}] 🚀 Démarrage")
            self.driver = self.downloader.create_driver()
            idle_count = 0
            max_idle_cycles = 20  # Attendre 20 cycles sans pages avant d'abandonner

            while True:
                page_info = self.downloader.backlog.get_next_pending_page(self.worker_id)
                if page_info is None:
                    idle_count += 1
                    if idle_count >= max_idle_cycles:
                        print(f"[Worker-{self.worker_id}] ⏹️ Arrêt - plus de pages après {idle_count} tentatives")
                        break
                    else:
                        # Attendre un peu et vérifier s'il y a des workers actifs
                        stats = self.downloader.backlog.get_stats()
                        if stats['in_progress'] == 0 and stats['pending'] == 0:
                            # Aucun worker actif et aucune page en attente = fin
                            print(f"[Worker-{self.worker_id}] ⏹️ Arrêt - aucune activité détectée")
                            break
                        print(f"[Worker-{self.worker_id}] ⏳ Attente de nouvelles pages... ({idle_count}/{max_idle_cycles})")
                        time.sleep(2)
                        continue

                # Reset idle counter si on trouve une page
                idle_count = 0
                success = self.downloader.process_page(self.driver, page_info.url, self.worker_id)

                if success:
                    self.downloader.backlog.mark_completed(page_info.url)
                    self.processed_count += 1
                    print(f"[Worker-{self.worker_id}] ✅ Page {self.processed_count} terminée")
                else:
                    self.downloader.backlog.mark_error(page_info.url, "Extraction failed")

                self.downloader.backlog.save_backlog()
                time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"[Worker-{self.worker_id}] 💥 Erreur critique: {e}")
        finally:
            if self.driver:
                self.driver.quit()
            print(f"[Worker-{self.worker_id}] 🏁 Arrêt - {self.processed_count} pages traitées")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Step01 - Téléchargement automatique de documentation")
    parser.add_argument("--start-url", action="append", required=True, help="URL de départ (peut être répété)")
    parser.add_argument("--base-url", help="URL de base pour filtrer les liens (défaut: domaine de la première start-url)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Répertoire de sortie")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="Nombre de workers")
    parser.add_argument("--resume", action="store_true", help="Reprendre téléchargement")

    args = parser.parse_args()

    # Déterminer l'URL de base automatiquement si non spécifiée
    if args.base_url:
        base_url = args.base_url
    else:
        # Utiliser le domaine de la première URL comme base
        from urllib.parse import urlparse
        parsed = urlparse(args.start_url[0])
        base_url = f"{parsed.scheme}://{parsed.netloc}"

    # Forcer le chemin dans le sous-dossier data
    final_output_dir = Path("data") / args.output
    final_output_dir.mkdir(parents=True, exist_ok=True)

    print("🌐 Step01 - Téléchargement Documentation Web")
    print("=" * 60)
    print(f"📁 Chemin demandé: {args.output}")
    print(f"📁 Chemin final: {final_output_dir}")
    print(f"🔧 Workers: {args.workers}")
    print(f"🌍 Base URL: {base_url}")
    print(f"📊 URLs de départ: {len(args.start_url)}")
    for i, url in enumerate(args.start_url, 1):
        print(f"   {i}. {url}")

    if args.resume:
        print("🔄 Mode reprise activé")

    # Initialisation
    downloader = WebDocumentDownloader(
        output_dir=str(final_output_dir),
        base_url=base_url,
        max_workers=args.workers
    )

    # Ajout des URLs de départ si pas en mode reprise
    if not args.resume:
        for url in args.start_url:
            downloader.backlog.add_page(url)
        downloader.backlog.save_backlog()

    # Lancement des workers
    try:
        workers = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for i in range(args.workers):
                worker = DownloadWorker(f"{i+1}", downloader)
                future = executor.submit(worker.start)
                workers.append((worker, future))

            # Monitoring
            while True:
                time.sleep(10)
                stats = downloader.backlog.get_stats()
                active_workers = sum(1 for _, future in workers if not future.done())

                print(f"\n📊 Statut: {stats['completed']} terminées, {stats['pending']} en attente, {stats['error']} erreurs")
                print(f"👥 Workers actifs: {active_workers}/{args.workers}")

                if active_workers == 0:
                    break

                downloader.backlog.save_backlog()

    except KeyboardInterrupt:
        print("\n⏹️ Interruption utilisateur")
        downloader.backlog.save_backlog()
        print("💾 État sauvegardé - utilisez --resume pour continuer")

    except Exception as e:
        print(f"\n💥 Erreur: {e}")
        downloader.backlog.save_backlog()
        raise

    finally:
        downloader.backlog.save_backlog()
        stats = downloader.backlog.get_stats()
        print(f"\n📈 Résultat final:")
        print(f"   • Total: {stats['total']} pages")
        print(f"   • Terminées: {stats['completed']}")
        print(f"   • Erreurs: {stats['error']}")
        print(f"   • Taux succès: {(stats['completed'] / max(stats['total'], 1) * 100):.1f}%")
        print(f"\n📁 Documentation téléchargée dans: {args.output}")
        print(f"▶️  Étape suivante: python step02_indexer.py {args.output}")


if __name__ == "__main__":
    main()