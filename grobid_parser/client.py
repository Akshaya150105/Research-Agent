from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import requests


class GROBIDClient:
   
    #Sends PDFs to GROBID and returns raw TEI XML.

    #Parameters
    #base_url : str -    URL of the running GROBID service. 
    #timeout : int -    Seconds to wait per request. 
    #max_retries : int -    Retry count on 503 (GROBID overloaded) or timeout.


    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        timeout: int = 120,      
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries


    def is_alive(self) -> bool:
        #Return True if GROBID service is reachable and ready.
        try:
            r = requests.get(f"{self.base_url}/api/isalive", timeout=5)
            return r.status_code == 200 and r.text.strip().lower() == "true"
        except Exception:
            return False

    def process_fulltext(
        self,
        pdf_path: Union[str, Path],
        consolidate_header: bool = True,
    ) -> Optional[str]:
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"[GROBID] Error: PDF not found: {pdf_path}")
            return None

        url = f"{self.base_url}/api/processFulltextDocument"
        params = {
            "consolidateHeader": "1" if consolidate_header else "0",
            "consolidateCitations": "1",  
            "includeRawCitations": "1",
            "includeRawAffiliations": "1",
            "teiCoordinates": "0",       
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                with open(pdf_path, "rb") as f:
                    response = requests.post(
                        url,
                        files={"input": (pdf_path.name, f, "application/pdf")},
                        data=params,
                        timeout=self.timeout,
                    )

                if response.status_code == 200:
                    return response.text

                elif response.status_code == 503:
                    wait = 10 * attempt
                    print(f"[GROBID] Service busy (503), waiting {wait}s "
                          f"(attempt {attempt}/{self.max_retries})...")
                    time.sleep(wait)

                else:
                    print(f"[GROBID] HTTP {response.status_code}: "
                          f"{response.text[:200]}")
                    return None

            except requests.exceptions.ConnectionError:
                print(
                    f"[GROBID] Cannot connect to {self.base_url}\n"
                    "  Is GROBID running? Start it with:\n"
                    "    cd grobid && ./gradlew run\n"
                    "  Then verify: curl http://localhost:8070/api/isalive"
                )
                return None

            except requests.exceptions.Timeout:
                print(f"[GROBID] Timeout on attempt {attempt}/{self.max_retries}")
                if attempt == self.max_retries:
                    return None

        return None