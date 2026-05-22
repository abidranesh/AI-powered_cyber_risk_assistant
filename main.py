"""
main.py — Orchestrator for TawasolPay Risk Engine (LangChain edition)
Works with the flat-script risk_engine.py (no classes).
"""

import os
import json
import pandas as pd
import requests
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import build_nist_vectorstore, get_nist_collection, retrieve_nist_control
from llm_summariser import setup_gemini, summarise_risk_with_nist

_HERE      = os.path.dirname(os.path.abspath(__file__))
NIST_CSV   = os.getenv("NIST_CSV", os.path.join(_HERE, "data", "nist_controls.csv"))
CHROMA_DIR = os.path.join(_HERE, "chroma_db")

# ── Scoring config (copied from your risk_engine.py) ─────────────────────────
SCORING_CONFIG = {
    "cvss_multiplier": 2.5,
    "default_cvss": 5.0,
    "internet_exposure_score": 25,
    "exploit_available_score": 20,
    "threat_actor_present_score": 15,
    "ransomware_association_score": 10,
    "cisa_kev_ransomware_score": 10,
    "compliance_critical_score": 8,
    "criticality_score": 7,
    "patch_unavailable_score": 5,
    "max_score": 100
}


def fetch_cisa_kev():
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    print("Fetching CISA KEV catalog...")
    try:
        resp = requests.get(url, timeout=15)
        kev_df = pd.DataFrame(resp.json()["vulnerabilities"])
        kev_cves = set(kev_df["cveID"].str.upper())
        kev_ransomware = set(
            kev_df[kev_df["knownRansomwareCampaignUse"].str.upper() == "KNOWN"]["cveID"].str.upper()
        )
        print(f"  Loaded {len(kev_cves)} KEV entries, {len(kev_ransomware)} ransomware-associated")
        return kev_cves, kev_ransomware
    except Exception as e:
        print(f"  Warning: could not fetch KEV ({e})")
        return set(), set()


def score_row(row, kev_ransomware):
    c = SCORING_CONFIG
    score = 0.0
    try:
        cvss = float(row.get("cvss", c["default_cvss"]))
    except (ValueError, TypeError):
        cvss = c["default_cvss"]
    score += cvss * c["cvss_multiplier"]

    if str(row.get("asset_exposure", "")).lower() == "internet" or \
       str(row.get("internet_exposed", "")).lower() == "yes":
        score += c["internet_exposure_score"]

    if str(row.get("exploit_available", "")).lower() == "yes":
        score += c["exploit_available_score"]

    if pd.notna(row.get("threat_actor")):
        score += c["threat_actor_present_score"]

    if str(row.get("ransomware_association", "")).lower() == "yes":
        score += c["ransomware_association_score"]

    if str(row.get("cve", "")).upper() in kev_ransomware:
        score += c["cisa_kev_ransomware_score"]

    compliance = str(row.get("compliance_scope", "")).lower()
    if "pci" in compliance or "gdpr" in compliance:
        score += c["compliance_critical_score"]

    if str(row.get("criticality", "")).lower() == "critical":
        score += c["criticality_score"]

    if str(row.get("patch_available", "")).lower() == "no":
        score += c["patch_unavailable_score"]

    return min(score, c["max_score"])


def build_top5():
    assets  = pd.read_csv("data/assets.csv")
    vulns   = pd.read_csv("data/vulnerabilities.csv")
    threat  = pd.read_csv("data/threat_intelligence.csv")
    biz     = pd.read_csv("data/business_services.csv")
    _       = pd.read_csv("data/remediation_guidance.csv")

    for df in [assets, vulns, threat, biz]:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    kev_cves, kev_ransomware = fetch_cisa_kev()

    # Aggregate threat intel — keep highest confidence per CVE
    if "matched_cve_or_control" in threat.columns:
        threat_agg = (
            threat.sort_values("confidence")
            .groupby("matched_cve_or_control", as_index=False)
            .last()
        )
    else:
        threat_agg = pd.DataFrame()

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

    merged["risk_score"]        = merged.apply(lambda r: score_row(r, kev_ransomware), axis=1)
    merged["in_kev"]            = merged["cve"].str.upper().isin(kev_cves)
    merged["in_kev_ransomware"] = merged["cve"].str.upper().isin(kev_ransomware)

    return (
        merged.sort_values("risk_score", ascending=False)
        .drop_duplicates(subset=["asset_id", "cve"])
        .head(5)
        .reset_index(drop=True)
    )


def main():
    # ── A: Build or load LangChain NIST vector store ──────────────────────
    if os.path.exists(CHROMA_DIR):
        print("Loading existing LangChain NIST vector store...")
        vectorstore = get_nist_collection()
    elif os.path.exists(NIST_CSV):
        print(f"Building LangChain NIST vector store from {NIST_CSV}...")
        vectorstore = build_nist_vectorstore(NIST_CSV)
    else:
        print(f"WARNING: {NIST_CSV} not found. NIST controls will be skipped.")
        vectorstore = None

    # ── B: Score and rank ─────────────────────────────────────────────────
    print("\nRunning risk engine...")
    top5 = build_top5()

    # Print terminal report (same format as your original risk_engine.py)
    print("\n" + "="*70)
    print("  TAWASALPAY — TOP 5 CYBER RISKS")
    print("="*70 + "\n")
    for i, row in top5.iterrows():
        rank = i + 1
        print(f"RISK #{rank}  (Score: {row.get('risk_score', 0):.0f}/100)")
        print(f"  Asset      : {row.get('asset_name', row.get('asset_id', 'Unknown'))}")
        print(f"  CVE        : {row.get('cve', 'N/A')}  (CVSS {row.get('cvss', '?')})")
        print(f"  Vuln       : {row.get('vulnerability_name', 'Unknown')}")
        print(f"  Service    : {row.get('business_service', 'Unknown')}")
        if pd.notna(row.get("threat_actor")):
            print(f"  Threat Actor: {row.get('threat_actor')} — {row.get('campaign_name', '')}")
        print()

    cols = ["asset_name", "cve", "cvss", "risk_score", "threat_actor", "campaign_name"]
    available = [c for c in cols if c in top5.columns]
    print(tabulate(top5[available], headers="keys", tablefmt="rounded_outline", showindex=False))

    # ── C: Set up LangChain LLM chain ─────────────────────────────────────
    chain = None
    if os.getenv("GEMINI_API_KEY"):
        print("\nSetting up LangChain LLM chain (Gemini 2.5 Flash)...")
        chain = setup_gemini()
    else:
        print("\nWARNING: GEMINI_API_KEY not set. AI summaries will be skipped.")

    # ── D: Enrich each risk with NIST + LLM ───────────────────────────────
    print("\n" + "="*70)
    print("  LANGCHAIN RAG + LLM ENRICHMENT")
    print("="*70)

    report_lines = ["# TawasolPay — Top 5 Cyber Risks\n"]

    for rank, row in top5.iterrows():
        risk_num = rank + 1
        risk     = row.where(pd.notna(row), None).to_dict()

        nist_control = {"control_id": "N/A", "control_name": "Skipped", "text": ""}
        if vectorstore is not None:
            query = (
                f"{risk.get('vulnerability_name', '')} "
                f"{risk.get('cve', '')} "
                f"{'internet-exposed' if str(risk.get('asset_exposure','')).lower() == 'internet' else 'internal'} "
                f"{'ransomware' if str(risk.get('ransomware_association','')).lower() == 'yes' else ''} "
                f"{risk.get('threat_actor', '') or ''} "
                f"{risk.get('business_service', '') or ''}"
            )
            nist_control = retrieve_nist_control(query, vectorstore)
            print(f"\nRisk #{risk_num} → NIST: {nist_control['control_id']} — {nist_control['control_name']}")

        llm_summary = "AI summary not available."
        if chain and nist_control["control_id"] != "N/A":
            print(f"          → Generating AI summary...")
            llm_summary = summarise_risk_with_nist(chain, risk, nist_control)

        block = f"""## Risk #{risk_num} — Score: {risk.get('risk_score', 0):.0f}/100

**Asset:** {risk.get('asset_name', risk.get('asset_id', 'Unknown'))}
**CVE:** {risk.get('cve', 'N/A')} (CVSS {risk.get('cvss', '?')})
**Vulnerability:** {risk.get('vulnerability_name', 'Unknown')}
**Business Service:** {risk.get('business_service', 'Unknown')}
**Threat Actor:** {risk.get('threat_actor', 'None matched')}
**NIST Control:** {nist_control['control_id']} — {nist_control['control_name']}

**AI Analysis:**
{llm_summary}

---
"""
        report_lines.append(block)
        print(block)

    # ── E: Write report ────────────────────────────────────────────────────
    with open("risk_report.md", "w") as f:
        f.write("\n".join(report_lines))
    print("\n✓ Report written to risk_report.md")


if __name__ == "__main__":
    main()
