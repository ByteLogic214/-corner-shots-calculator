import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class StatsAPIClient:
    def __init__(self):
        self.api_key = os.getenv("THESTATSAPI_KEY")
        self.base_url = "https://api.thestatsapi.com/v1/football"
        
        # CORRECCIÓN CRÍTICA: Autenticación mediante Bearer token en Authorization header
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        })

    def get_competitions(self) -> list:
        res = self.session.get(f"{self.base_url}/competitions")
        res.raise_for_status()
        return res.json().get("competitions", [])

    def search_team_id(self, team_name: str) -> int:
        res = self.session.get(f"{self.base_url}/teams/search", params={"q": team_name})
        res.raise_for_status()
        results = res.json().get("results", [])
        if not results:
            raise ValueError(f"Equipo no encontrado en TheStatsAPI: {team_name}")
        return results[0]["id"]

    def _fetch_single_match_stats(self, match: dict) -> dict:
        """Función auxiliar para descargar estadísticas individuales."""
        match_id = match.get("id")
        if not match_id:
            return match
            
        try:
            stats_res = self.session.get(f"{self.base_url}/matches/{match_id}/stats", timeout=5)
            if stats_res.status_code == 200:
                match["detailed_stats"] = stats_res.json().get("stats", {})
        except requests.RequestException:
            match["detailed_stats"] = {} # Fallback seguro si falla la red
            
        return match

    def get_last_10_matches_stats(self, team_id: int) -> list:
        """Obtiene los últimos 10 partidos finalizados optimizando con concurrencia."""
        res = self.session.get(
            f"{self.base_url}/teams/{team_id}/matches", 
            params={"limit": 10, "status": "FINISHED"}
        )
        res.raise_for_status()
        matches = res.json().get("matches", [])
        
        # OPTIMIZACIÓN: Descarga paralela utilizando un pool de hilos (Multi-threading)
        detailed_matches = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Lanzamos las 10 llamadas HTTP en paralelo
            futures = {executor.submit(self._fetch_single_match_stats, m): m for m in matches}
            
            for future in as_completed(futures):
                detailed_matches.append(future.result())
                
        return detailed_matches

    def get_match_odds(self, match_id: str) -> dict:
        if not match_id:
            return {}
        res = self.session.get(f"{self.base_url}/matches/{match_id}/odds")
        return res.json().get("odds", {}) if res.status_code == 200 else {}

    def get_player_stats(self, player_id: str) -> dict:
        res = self.session.get(f"{self.base_url}/players/{player_id}/stats")
        return res.json().get("stats", {}) if res.status_code == 200 else {}
