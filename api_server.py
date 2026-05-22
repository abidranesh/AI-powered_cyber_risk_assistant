"""
TawasolPay Risk Engine — FastAPI backend
Run: uvicorn api_server:app --reload --port 8000
Then open index.html in your browser (or serve it with: python -m http.server 3000)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import json, os, sys, threading

# ── Add parent directory to path so we can import your existing modules ──
# If api_server.py is in a subfolder, adjust this path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Import your existing risk engine pieces ──
# These come from risk_engine.py (your existing file)
try:
    from risk_engine import DataLoader, ThreatIntelProcessor, RiskScorer, SCORING_CONFIG
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("WARNING: risk_engine.py not found. Returning empty results.")

# ── Import your new RAG + LLM modules (from Step 3 / 4 of the guide) ──
try:
    from rag_pipeline import get_nist_collection, retrieve_nist_control, build_nist_vectorstore, CHROMA_DIR
    from llm_summariser import setup_gemini, summarise_risk_with_nist
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    print("WARNING: rag_pipeline.py or llm_summariser.py not found. NIST + LLM features disabled.")

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="TawasolPay Risk API", version="1.0")

# ── CORS: allow the frontend (opened as a local file or on port 3000) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Eagerly-loaded objects (built on startup) ─────────────────────────────
_nist_collection = None
_gemini_model = None

def _load_nist_background():
    """Runs in a daemon thread at startup. Sets _nist_collection when ready."""
    global _nist_collection
    if not HAS_RAG:
        return
    try:
        _here    = os.path.dirname(os.path.abspath(__file__))
        nist_csv = os.getenv("NIST_CSV", os.path.join(_here, "data", "nist_controls.csv"))
        if os.path.exists(CHROMA_DIR):
            _nist_collection = get_nist_collection()
        elif os.path.exists(nist_csv):
            print(f"chroma_db not found — building vector store from {nist_csv} ...")
            _nist_collection = build_nist_vectorstore(nist_csv)
        else:
            print(f"WARNING: neither {CHROMA_DIR} nor {nist_csv} found. NIST RAG disabled.")
    except Exception as e:
        print(f"NIST vector store load failed: {e}")

def get_gemini():
    global _gemini_model
    if HAS_RAG and _gemini_model is None:
        try:
            _gemini_model = setup_gemini()
        except Exception as e:
            print(f"Gemini setup failed: {e}")
    return _gemini_model

@app.on_event("startup")
async def startup():
    print("Starting up — launching NIST vector store load in background thread...")
    get_gemini()  # fast — just builds the chain object, no API calls
    threading.Thread(target=_load_nist_background, daemon=True).start()
    print("Startup complete. Requests will return risk scores immediately; NIST enrichment added once vector store is ready.")


# ── /api/risks endpoint ──────────────────────────────────────────────────
@app.get("/api/risks")
def get_risks():
    if not HAS_ENGINE:
        return {"risks": [], "error": "risk_engine.py not available"}

    # 1. Load and score data (your existing logic)
    _here = os.path.dirname(os.path.abspath(__file__))
    try:
        assets   = DataLoader.load_csv(os.path.join(_here, "data", "assets.csv"))
        vulns    = DataLoader.load_csv(os.path.join(_here, "data", "vulnerabilities.csv"))
        threat   = DataLoader.load_csv(os.path.join(_here, "data", "threat_intelligence.csv"))
        biz      = DataLoader.load_csv(os.path.join(_here, "data", "business_services.csv"))
        kev_cves, kev_ransomware = DataLoader.fetch_cisa_kev()
    except Exception as e:
        return {"risks": [], "error": str(e)}

    threat_agg = ThreatIntelProcessor.aggregate_by_cve(threat)

    merged = vulns.merge(assets, on="asset_id", how="left")
    merged = merged.merge(biz, on="business_service", how="left")

    if not threat_agg.empty:
        threat_cols = ["matched_cve_or_control", "threat_actor", "campaign_name",
                       "ransomware_association", "exploit_maturity", "active_last_seen"]
        available = [c for c in threat_cols if c in threat_agg.columns]
        merged = merged.merge(
            threat_agg[available],
            left_on="cve", right_on="matched_cve_or_control", how="left"
        )

    scorer = RiskScorer(SCORING_CONFIG, kev_ransomware)
    merged["risk_score"] = merged.apply(scorer.score_row, axis=1)
    merged["in_kev"]     = merged["cve"].str.upper().isin(kev_cves)

    top5 = (
        merged.sort_values("risk_score", ascending=False)
        .drop_duplicates(subset=["asset_id", "cve"])
        .head(5)
        .reset_index(drop=True)
    )

    # 2. Attach RAG + LLM enrichment if ready (non-blocking — background thread loads this)
    collection    = _nist_collection
    gemini_model  = get_gemini()

    results = []
    for _, row in top5.iterrows():
        risk = row.where(pd.notna(row), None).to_dict()

        # Convert numpy types to plain Python for JSON serialisation
        for k, v in risk.items():
            if hasattr(v, "item"):       # numpy scalar
                risk[k] = v.item()
            elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
                risk[k] = None

        nist_result = None
        llm_text    = None

        if collection:
            query = (
                f"{risk.get('vulnerability_name','')} "
                f"{risk.get('cve','')} "
                f"{'internet-exposed' if str(risk.get('asset_exposure','')).lower()=='internet' else 'internal'} "
                f"{'ransomware' if str(risk.get('ransomware_association','')).lower()=='yes' else ''} "
                f"{risk.get('threat_actor','') or ''} "
                f"{risk.get('business_service','') or ''}"
            )
            nist_result = retrieve_nist_control(query, collection)

        if gemini_model and nist_result:
            try:
                llm_text = summarise_risk_with_nist(gemini_model, risk, nist_result)
            except Exception as e:
                llm_text = f"LLM error: {e}"

        risk["nist_control"] = nist_result
        risk["llm_analysis"] = llm_text
        results.append(risk)

    return {"risks": results}


# ── Health check ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": HAS_ENGINE,
        "rag": HAS_RAG,
        "groq_key_set": bool(os.getenv("GROQ_API_KEY"))
    }


# ── Serve the frontend at / ──────────────────────────────────────────────
# Must come after all /api routes so API routes are matched first.
# html=True means GET / returns index.html automatically.
_static_dir = os.path.dirname(os.path.abspath(__file__))
try:
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
except Exception as e:
    print(f"WARNING: could not mount static files from {_static_dir}: {e}")
