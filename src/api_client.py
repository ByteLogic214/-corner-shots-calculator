import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class StatsAPIClient:
    def __init__(self):
        self.api_key = os.getenv("THESTATSAPI_KEY")
        self.base_url = "https://thestatsapi.com"
        
        # Autenticación oficial mediante Bearer token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        })

    def get_competitions(self) -> list:
        res = self.session.get(f"{self.base_url}/competitions")
        res.raise_for_status()
        return res.json().get("data", [])

    def search_team_id(self, team_name: str) -> int:
        """Busca el ID de un equipo usando parámetros reales filtrando coincidencias."""
        res = self.session.get(
            f"{self.base_url}/teams",
            params={"search": team_name, "per_page": 100}
        )
        res.raise_for_status()

        teams = res.json().get("data", [])
        if not teams:
            raise ValueError(f"La API no devolvió ningún resultado para: {team_name}")

        # Intento 1: Coincidencia exacta
        for team in teams:
            if team["name"].lower() == team_name.lower():
                return team["id"]

        # Intento 2: Fallback al primer resultado parcial de la búsqueda
        return teams[0]["id"]

    def _fetch_single_match_stats(self, match: dict) -> dict:
        """Descarga las estadísticas avanzadas de un partido individual."""
        match_id = match.get("id")
        if not match_id:
            return match
            
        try:
            stats_res = self.session.get(f"{self.base_url}/matches/{match_id}/stats", timeout=5)
            if stats_res.status_code == 200:
                match["detailed_stats"] = stats_res.json().get("data", {})
        except requests.RequestException:
            match["detailed_stats"] = {}
            
        return match

    def get_last_10_matches_stats(self, team_id: int) -> list:
        """Obtiene el historial optimizando la descarga mediante subprocesos en paralelo."""
        res = self.session.get(
            f"{self.base_url}/teams/{team_id}/matches", 
            params={"limit": 10, "status": "FINISHED"}
        )
        res.raise_for_status()
        matches = res.json().get("data", [])
        
        detailed_matches = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._fetch_single_match_stats, m): m for m in matches}
            for future in as_completed(futures):
                detailed_matches.append(future.result())
                
        return detailed_matches

    def get_match_odds(self, match_id: str) -> dict:
        if not match_id:
            return {}
        res = self.session.get(f"{self.base_url}/matches/{match_id}/odds")
        return res.json().get("data", {}) if res.status_code == 200 else {}
