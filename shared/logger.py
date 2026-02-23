# ============================================================================
# shared/logger.py
# ============================================================================
"""
Centralised logger for all pipeline steps.

Usage:
    from shared.logger import logger
    logger.info("Step 2 prefilter loaded.")
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("recovery")
