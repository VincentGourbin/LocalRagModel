#!/usr/bin/env python3
"""
Step 03 - Générateur SSL optionnel pour serveur MCP HTTPS
Génère des certificats auto-signés pour l'intégration Claude Desktop
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_ssl_certificates():
    """Génère des certificats SSL auto-signés pour localhost"""
    
    print("🔒 Génération des certificats SSL pour LocalRAG MCP")
    print("=" * 50)
    
    # Répertoire pour les certificats
    ssl_dir = Path("ssl_certs")
    ssl_dir.mkdir(exist_ok=True)
    
    key_file = ssl_dir / "localhost.key"
    cert_file = ssl_dir / "localhost.crt"
    
    # Vérifier si openssl est disponible
    try:
        result = subprocess.run(["openssl", "version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ OpenSSL trouvé: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ OpenSSL non trouvé. Installation requise:")
        print("   macOS: brew install openssl")
        print("   Ubuntu: sudo apt-get install openssl")
        print("   Windows: télécharger depuis https://slproweb.com/products/Win32OpenSSL.html")
        return False
    
    # Configuration du certificat
    config_content = """[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = Local
L = Local
O = LocalRAG
CN = localhost

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = 127.0.0.1
IP.1 = 127.0.0.1
IP.2 = ::1
"""
    
    config_file = ssl_dir / "localhost.conf"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("📝 Configuration SSL créée")
    
    # Générer la clé privée
    try:
        subprocess.run([
            "openssl", "genrsa", 
            "-out", str(key_file), 
            "2048"
        ], check=True, capture_output=True)
        print("🔑 Clé privée générée")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur génération clé: {e}")
        return False
    
    # Générer le certificat
    try:
        subprocess.run([
            "openssl", "req",
            "-new", "-x509",
            "-key", str(key_file),
            "-out", str(cert_file),
            "-days", "365",
            "-config", str(config_file)
        ], check=True, capture_output=True)
        print("📜 Certificat généré")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur génération certificat: {e}")
        return False
    
    # Vérifier les certificats
    try:
        result = subprocess.run([
            "openssl", "x509", "-in", str(cert_file), 
            "-text", "-noout"
        ], capture_output=True, text=True, check=True)
        print("✅ Certificats validés")
    except subprocess.CalledProcessError:
        print("⚠️ Avertissement: validation certificat échoué")
    
    # Instructions d'utilisation
    abs_key = key_file.absolute()
    abs_cert = cert_file.absolute()
    
    print("\n🎯 Certificats SSL générés avec succès!")
    print(f"📁 Répertoire: {ssl_dir.absolute()}")
    print(f"🔑 Clé privée: {abs_key}")
    print(f"📜 Certificat: {abs_cert}")
    
    print("\n🚀 Pour utiliser HTTPS avec LocalRAG:")
    print(f"export SSL_KEYFILE='{abs_key}'")
    print(f"export SSL_CERTFILE='{abs_cert}'")
    print("python step03_chatbot.py")
    
    print("\n🔧 Configuration Claude Desktop (claude_desktop_config.json):")
    print(f"""{{
  "mcpServers": {{
    "localrag": {{
      "command": "python",
      "args": ["{Path.cwd() / 'step03_chatbot.py'}"],
      "env": {{
        "SSL_KEYFILE": "{abs_key}",
        "SSL_CERTFILE": "{abs_cert}",
        "PYTHONPATH": "{Path.cwd()}"
      }}
    }}
  }}
}}""")
    
    print("\n⚠️ Note: Certificat auto-signé - votre navigateur affichera un avertissement")
    print("   Cliquez sur 'Avancé' puis 'Continuer vers localhost'")
    
    # Nettoyer le fichier de config temporaire
    config_file.unlink()
    
    return True

def main():
    """Point d'entrée principal"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Générateur de certificats SSL pour LocalRAG MCP HTTPS")
        print("Usage: python generate_ssl_certs.py")
        print("\nGénère des certificats auto-signés pour localhost")
        return 0
    
    success = generate_ssl_certificates()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())