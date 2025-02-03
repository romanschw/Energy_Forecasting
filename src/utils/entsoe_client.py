import logging
import requests
import os
from typing import Optional, Dict, Any
from datetime import datetime
from eic_mapping import Area
from dotenv import load_dotenv

# Charger le fichier .env
load_dotenv()


# Génère les logs
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
base_url = "https://web-api.tp.entsoe.eu/api"

class ENTSOEClient:
    def __init__(self, api_key: Optional[str] = None,
                 session: Optional[requests.Session]=None,
                 ):
        self.api_key = api_key or os.getenv("ENTSOE_API_KEY")
        logger.info("API key %s", os.getenv("ENTSOE_API_KEY"))
        if not self.api_key:
            raise ValueError("A valid API key is necessary.")
        if session is None:
            self.session = requests.Session() # créer une session persistante pour améliorer la performance des requêtes au mêmes Hosts
        self.last_session = datetime.now()
    
    def _build_url(self, endpoint: str)->str:
        return f"{base_url}/{endpoint}"

    def _request(self, params: Dict[str, Any]) -> requests.Response :
        params.update({"securityToken": self.api_key})
        logger.info("requesting with params %s", params)
        try:
            response = self.session.get(url=base_url, params=params, timeout=10)
        except requests.RequestException as e:
            logger.error("Error during request: %s", e)
            raise #reraise the error
        return response

    def get_day_ahead_prices(self, start: str, end: str, country_code: str):
        """
        start : Pattern yyyyMMddHHmm e.g. 201601010000
        end : Pattern yyyyMMddHHmm e.g. 201601010000
        country_code : e.g. 'FR'
        """
        eic_code = Area.from_string(country_code)
        logger.info("Found EIC code: %s", eic_code)

        params = {"documentType": "A44",
                  "in_domain": eic_code,
                  "out_domain": eic_code,
                  "periodStart": start,
                  "periodEnd": end,
                  "offset": 0}
        
        response = self._request(params=params)
        return response.text
    
if __name__ == "__main__":
    client = ENTSOEClient()
    try:
        prices = client.get_day_ahead_prices(start="202301010000", end="202302010000", country_code="FR")
        print(prices)
    except requests.RequestException as e:
        logger.info("%s", e)
