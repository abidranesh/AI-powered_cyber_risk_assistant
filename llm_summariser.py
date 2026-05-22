"""
llm_summariser.py  —  LangChain-powered LLM summarisation (Groq)
=================================================================
Uses LangChain's LCEL pipe syntax:
    PromptTemplate | ChatGroq | StrOutputParser

PUBLIC API (unchanged):
    setup_gemini()                              -> chain object
    summarise_risk_with_nist(chain, risk, nist) -> str
"""

import os
import logging

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

log = logging.getLogger(__name__)

# ── Prompt template ────────────────────────────────────────────────────────
RISK_SUMMARY_TEMPLATE = PromptTemplate(
    input_variables=[
        "asset_name", "cve", "cvss", "vulnerability_name",
        "business_service", "threat_actor", "internet_exposed",
        "ransomware_association", "nist_control_id",
        "nist_control_name", "nist_control_text"
    ],
    template="""You are a cybersecurity analyst writing a brief for a CISO.

RISK DETAILS:
- Asset: {asset_name}
- CVE: {cve} (CVSS {cvss})
- Vulnerability: {vulnerability_name}
- Business service: {business_service}
- Threat actor: {threat_actor}
- Internet exposed: {internet_exposed}
- Ransomware associated: {ransomware_association}

NIST SP 800-53 CONTROL RETRIEVED:
Control: {nist_control_id} — {nist_control_name}
Full text: {nist_control_text}

Write 3-4 sentences explaining:
1. Why this risk is dangerous (use the specific context above, not generic advice)
2. What NIST control {nist_control_id} recommends in plain English, not jargon
3. The single most important immediate action

Be specific. Do not mention CVSS scores as numbers. Do not use the phrase 'it is important to'."""
)


# ── Public API ─────────────────────────────────────────────────────────────

def setup_gemini():
    """
    Builds and returns a LangChain LCEL chain backed by Groq (llama-3.1-8b-instant).
    Name kept as setup_gemini() so api_server.py and main.py need no changes.
    Requires GROQ_API_KEY in environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file or Railway environment variables.\n"
            "Get a free key at: https://console.groq.com"
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=512,
    )

    chain = RISK_SUMMARY_TEMPLATE | llm | StrOutputParser()

    log.info("LangChain LCEL chain (Groq llama-3.1-8b-instant) ready")
    return chain


def summarise_risk_with_nist(chain, risk: dict, nist_control: dict) -> str:
    """
    Invoke the LCEL chain with risk + NIST data.
    chain.invoke(values) fills the template, calls Groq, returns plain string.
    """
    values = {
        "asset_name"            : risk.get("asset_name") or risk.get("asset_id") or "Unknown",
        "cve"                   : risk.get("cve") or "N/A",
        "cvss"                  : risk.get("cvss") or "?",
        "vulnerability_name"    : risk.get("vulnerability_name") or "Unknown vulnerability",
        "business_service"      : risk.get("business_service") or "Unknown",
        "threat_actor"          : risk.get("threat_actor") or "None identified",
        "internet_exposed"      : str(risk.get("asset_exposure") or risk.get("internet_exposed") or "Unknown"),
        "ransomware_association": str(risk.get("ransomware_association") or "No"),
        "nist_control_id"       : nist_control.get("control_id") or "N/A",
        "nist_control_name"     : nist_control.get("control_name") or "N/A",
        "nist_control_text"     : (nist_control.get("text") or "")[:1500],
    }

    try:
        return chain.invoke(values).strip()
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return f"LLM summarisation unavailable: {e}"
