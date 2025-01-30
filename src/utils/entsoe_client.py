import logging
import requests
from typing import Optional

# Génère les logs
logger = logging.getLogger(__name__)
# Handler pour envoyer les messages de log
handler = logging.StreamHandler()
# Formater les messages de log pour être lisible
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Ajout du formatter au handler
handler.setFormatter(formatter)
# Ajout du handler au logger
logger.addHandler(handler)

class ENTSOEClient:
    def __init__(self, api_key: str,
                 session: Optional[requests.Session]=None,
                 ):
        self.session = requests.Session # créer une session persistante pour améliorer la performance des requêtes au mêmes Hosts
        self.session.headers.update({"Content-Type": "application/json"}) #Evite de repréciser à chaque fois
        self.api_key = api_key
        self.rate_limit = 100
        self.last
