"""Etherscan API client for fetching Ethereum transaction data."""

import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = "1"  # Ethereum mainnet
RATE_LIMIT_DELAY = 0.25  # 4 requests per second (free tier allows 5/sec, leave margin)
MAX_RETRIES = 3
TIMEOUT = 60  # seconds

logger = logging.getLogger(__name__)


class EtherscanClient:
    """Fetch transaction data from the Etherscan API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ETHERSCAN_API_KEY not found. "
                "Set it in .env or pass it to EtherscanClient()."
            )
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, params: dict[str, str]) -> Any:
        """Make a rate-limited GET request to Etherscan API with retry."""
        params["apikey"] = self.api_key
        params["chainid"] = CHAIN_ID

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()
            try:
                response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "0" and data.get("message") != "No transactions found":
                    raise Exception(f"Etherscan API error: {data.get('result', 'Unknown error')}")

                return data.get("result", [])

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.warning(f"Request failed (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Request failed after {MAX_RETRIES} attempts: {e}")
                    raise

    def get_normal_transactions(
        self, address: str, start_block: int = 0, end_block: int = 99999999
    ) -> list[dict]:
        """Fetch normal (ETH transfer) transactions for an address."""
        return self._get({
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": str(start_block),
            "endblock": str(end_block),
            "sort": "asc",
        })

    def get_internal_transactions(
        self, address: str, start_block: int = 0, end_block: int = 99999999
    ) -> list[dict]:
        """Fetch internal transactions for an address."""
        return self._get({
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "startblock": str(start_block),
            "endblock": str(end_block),
            "sort": "asc",
        })

    def get_erc20_transfers(
        self, address: str, start_block: int = 0, end_block: int = 99999999
    ) -> list[dict]:
        """Fetch ERC-20 token transfer events for an address."""
        return self._get({
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": str(start_block),
            "endblock": str(end_block),
            "sort": "asc",
        })

    def get_balance(self, address: str) -> float:
        """Get ETH balance for an address (in ETH, not Wei)."""
        result = self._get({
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest",
        })
        return int(result) / 1e18

    def get_all_transactions(self, address: str) -> dict[str, list[dict]]:
        """Fetch all transaction types for an address.

        Returns a dict with keys: normal, internal, erc20.
        Each key is fetched independently, so a failure in one
        does not block the others.
        """
        result = {}
        for txn_type, fetcher in [
            ("normal", self.get_normal_transactions),
            ("internal", self.get_internal_transactions),
            ("erc20", self.get_erc20_transfers),
        ]:
            try:
                result[txn_type] = fetcher(address)
            except Exception as e:
                logger.warning(f"Failed to fetch {txn_type} transactions: {e}")
                result[txn_type] = []

        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = EtherscanClient()
    test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

    print(f"Fetching data for {test_address[:10]}...")
    balance = client.get_balance(test_address)
    print(f"Balance: {balance:.4f} ETH")

    txns = client.get_normal_transactions(test_address)
    if isinstance(txns, list):
        print(f"Normal transactions: {len(txns)}")
        if txns:
            first = txns[0]
            print(f"First txn: {first.get('hash', 'N/A')[:20]}... "
                  f"value={int(first.get('value', 0)) / 1e18:.4f} ETH")
    else:
        print(f"Result: {txns}")
