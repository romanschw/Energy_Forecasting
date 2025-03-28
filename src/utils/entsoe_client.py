import logging
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import os
import sys
import pandas as pd
from datetime import timedelta
from xml.etree import ElementTree as ET
sys.path.append('/workspaces/ds_project_1/src/utils')

from typing import Optional, Dict, Any
from datetime import datetime
import time

from eic_mapping import Area, lookup_area

from dotenv import load_dotenv
from entsoe_parser import parse_prices, parse_loads, parse_generation

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

        if not self.api_key:
            raise ValueError("A valid API key is necessary.")

        self.session = session or requests.Session() # créer une session persistante pour améliorer la performance des requêtes au mêmes Hosts
            # Add retry logic
        retry_strategy = Retry(
            total=5,  # Maximum number of retries
            backoff_factor=1,  # Exponential backoff factor
            status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET"]  # Retry only on GET requests
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Add a delay between requests
        self.request_delay = 5  # Delay in seconds between requests
        self.last_session = datetime.now()
    
    def _build_url(self, endpoint: str)->str:
        return f"{base_url}/{endpoint}"

    def _request(self, params: Dict[str, Any]) -> requests.Response :

        params.update({"securityToken": self.api_key})
        logger.info("requesting with params %s", params)

        if hasattr(self,"last_request_time"):
            delay = (datetime.now() - self.last_request_time)
            if delay < timedelta(seconds=self.request_delay):
                sleep_time = (timedelta(seconds=self.request_delay) - delay)
                time.sleep(sleep_time.total_seconds())
        try:
            response = self.session.get(url=base_url, params=params, timeout=20)
        except requests.RequestException as e:
            logger.error("Error during request: %s", e)
            raise #reraise the exception
        self.last_request_time = datetime.now()
        return response
    
    def _parse_xml_response(self, xml_data: str) -> pd.DataFrame:
        """Parse XML using entsoe-py's parser."""
        try:
            # Use existing parser for prices
            parsed_data = parse_prices(xml_data)
            return parsed_data['60min'].to_frame(name="Price")  # Return hourly prices
        except Exception as e:
            logger.error("Parsing failed: %s", e)
            raise
    
    def get_multi_years(self, start: str, end:str, country_code: str, data_type: str)->pd.DataFrame:

        start_date = datetime.strptime(start, "%Y%m%d%H%M")
        end_date = datetime.strptime(end, "%Y%m%d%H%M")
        current_date = start_date
        multi_years_data =  pd.DataFrame()
        try:
            while current_date < end_date:
                current_end = min(current_date + timedelta(days=365), end_date)

                if data_type=="prices":
                    data = self.get_day_ahead_prices(start_date.strftime("%Y%m%d%H%M"), end=current_end.strftime("%Y%m%d%H%M"), country_code=country_code)
                elif data_type=="loads":
                    data = self.get_system_load_forecast(start=start_date.strftime("%Y%m%d%H%M"), end=current_end.strftime("%Y%m%d%H%M"), country_code=country_code)
                elif data_type=="generation":
                    data = self.get_system_load_forecast(start=start_date.strftime("%Y%m%d%H%M"), end=current_end.strftime("%Y%m%d%H%M"), country_code=country_code)
                multi_years_data = pd.concat([multi_years_data, data])
        except Exception as e:
            logger.error(f"Failed to retrieve data for {current_date}-{current_end} with error: %s", e)
        return multi_years_data

    def get_day_ahead_prices(self, start: str, end: str, country_code: str):
        """
        start : Pattern yyyyMMddHHmm e.g. 201601010000
        end : Pattern yyyyMMddHHmm e.g. 201601010000
        country_code : e.g. 'FR'
        """
        eic_code = lookup_area(country_code)
        logger.info("Found EIC code: %s", eic_code)

        params = {"documentType": "A44",
                  "in_domain": eic_code,
                  "out_domain": eic_code,
                  "periodStart": start,
                  "periodEnd": end,
                  "offset": 0}
        
        response = self._request(params=params)
        return self._parse_xml_response(response.text)
    
    def get_system_load_forecast(self, start: str, end: str, country_code: str) -> pd.DataFrame:
        eic_code = lookup_area(country_code)
        logger.info("Found cound EIC code: %s", eic_code)

        params = {"documentType": "A44",
                  "in_domain": eic_code,
                  "out_domain": eic_code,
                  "periodStart": start,
                  "periodEnd": end,
                  "offset": 0}
        
        response = self._request(params=params)
        return parse_loads(response.text)
    
    def get_renewable_energy_forecast(self, start: str, end: str, country_code: str) -> pd.DataFrame:
        """
        Retrieve renewable energy generation forecast for a given time range and country.
        
        Args:
            start: Start time in the format 'yyyyMMddHHmm' (e.g., '202301010000').
            end: End time in the format 'yyyyMMddHHmm' (e.g., '202301010000').
            country_code: Country code (e.g., 'FR').
        
        Returns:
            A pandas DataFrame containing the renewable energy generation forecast.
        """
        eic_code = lookup_area(country_code)
        logger.info("Found EIC code: %s", eic_code)

        params = {
            "documentType": "A69",
            "in_domain": eic_code,
            "out_domain": eic_code,
            "periodStart": start,
            "periodEnd": end,
            "offset": 0
        }

        response = self._request(params=params)
        return parse_generation(response.text)

if __name__ == "__main__":
    client = ENTSOEClient()
    try:
        prices = client.get_day_ahead_prices(start="202301010000", end="202302010000", country_code="FR")
        print(prices)
    except requests.RequestException as e:
        logger.info("%s", e)
