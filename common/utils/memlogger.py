# app/utils/memlogger.py
import os
import psutil
import logging

def log_memory(tag=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logging.info(f"[MEMORY] {tag} - RSS memory: {mem_mb:.2f} MB")
