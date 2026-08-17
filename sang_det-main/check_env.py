"""Verification script to test .env connections: Supabase PostgreSQL, Supabase Storage, and NVIDIA NIM API."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv(Path(__file__).parent / ".env")

results = []

# 1. Check PostgreSQL Database Connection
db_url = os.environ.get("DATABASE_URL")
if not db_url:
    results.append(("Database", False, "DATABASE_URL is missing in .env"))
else:
    try:
        import psycopg2
        conn = psycopg2.connect(db_url, connect_timeout=10)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            ver = cur.fetchone()[0]
            # Check if tables exist
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            tables = [r[0] for r in cur.fetchall()]
        conn.close()
        results.append(("Database", True, f"Connected to PostgreSQL! Tables found: {tables}"))
    except Exception as exc:
        results.append(("Database", False, f"PostgreSQL connection failed: {exc}"))

# 2. Check Supabase Storage
sb_url = os.environ.get("SUPABASE_URL")
sb_key = os.environ.get("SUPABASE_KEY")
sb_bucket = os.environ.get("SUPABASE_BUCKET", "plates")
if not sb_url or not sb_key:
    results.append(("Supabase Storage", False, "SUPABASE_URL or SUPABASE_KEY missing in .env"))
else:
    try:
        import httpx
        headers = {
            "apikey": sb_key,
            "Authorization": f"Bearer {sb_key}",
        }
        # Check buckets endpoint
        resp = httpx.get(f"{sb_url.rstrip('/')}/storage/v1/bucket", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            buckets = [b.get("name") for b in resp.json()]
            bucket_found = sb_bucket in buckets
            results.append(("Supabase Storage", True, f"Connected to Storage! Buckets: {buckets} (target bucket '{sb_bucket}' present: {bucket_found})"))
        else:
            results.append(("Supabase Storage", False, f"Storage check returned HTTP {resp.status_code}: {resp.text}"))
    except Exception as exc:
        results.append(("Supabase Storage", False, f"Storage connection error: {exc}"))

# 3. Check NVIDIA NIM API Key
nim_key = os.environ.get("NVIDIA_API_KEY")
if not nim_key:
    results.append(("NVIDIA NIM API", False, "NVIDIA_API_KEY missing in .env"))
else:
    try:
        import httpx
        headers = {
            "Authorization": f"Bearer {nim_key}",
            "Accept": "application/json",
        }
        resp = httpx.get("https://integrate.api.nvidia.com/v1/models", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            results.append(("NVIDIA NIM API", True, "API Key is valid and active on NVIDIA NIM!"))
        else:
            results.append(("NVIDIA NIM API", False, f"NVIDIA NIM returned HTTP {resp.status_code}: {resp.text[:200]}"))
    except Exception as exc:
        results.append(("NVIDIA NIM API", False, f"NVIDIA NIM error: {exc}"))

print("=" * 60)
print("  ENV VALIDATION RESULTS")
print("=" * 60)
for name, ok, msg in results:
    status = "[PASS]" if ok else "[FAIL]"
    print(f"{status} {name}: {msg}")
print("=" * 60)
