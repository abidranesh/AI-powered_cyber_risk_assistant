import pandas as pd
import requests
from tabulate import tabulate

# ── Scoring weights (used by RiskScorer and api_server.py) ───────────────────
SCORING_CONFIG = {
    "cvss_multiplier":            2.5,
    "default_cvss":               5.0,
    "internet_exposure_score":   25,
    "exploit_available_score":   20,
    "threat_actor_present_score": 15,
    "ransomware_association_score": 10,
    "cisa_kev_ransomware_score":  10,
    "exploit_maturity_score":      8,  # weaponized/active-exploitation from threat intel
    "compliance_critical_score":   8,
    "criticality_score":           7,
    "patch_unavailable_score":     5,
    "max_score":                 100,
}


class DataLoader:
    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        # Normalize CVE/control identifier values so joins never silently drop rows
        # due to case or whitespace differences (e.g. "cve-2024-1234" vs "CVE-2024-1234")
        for col in ("cve", "matched_cve_or_control"):
            if col in df.columns:
                df[col] = df[col].str.strip().str.upper()
        return df

    @staticmethod
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
            print(f"  Warning: could not fetch KEV ({e}). Proceeding without it.")
            return set(), set()


class ThreatIntelProcessor:
    @staticmethod
    def aggregate_by_cve(threat: pd.DataFrame) -> pd.DataFrame:
        if "matched_cve_or_control" not in threat.columns:
            return pd.DataFrame()
        return (
            threat.sort_values("confidence")
            .groupby("matched_cve_or_control", as_index=False)
            .last()
        )

    @staticmethod
    def log_unmatched(threat_agg: pd.DataFrame, vuln_cves: set) -> None:
        """Warn about threat-intel keys that won't join to any vulnerability row."""
        if threat_agg.empty or "matched_cve_or_control" not in threat_agg.columns:
            return
        unmatched = [
            k for k in threat_agg["matched_cve_or_control"]
            if pd.notna(k) and k not in vuln_cves
        ]
        if unmatched:
            print(
                f"WARNING: {len(unmatched)} threat-intel key(s) had no matching vuln CVE "
                f"and will be ignored in scoring: {unmatched}",
                flush=True
            )


class RiskScorer:
    def __init__(self, config: dict, kev_ransomware: set):
        self.config = config
        self.kev_ransomware = kev_ransomware

    def score_row(self, row) -> float:
        c = self.config
        score = 0.0
        try:
            cvss = float(row.get("cvss", c["default_cvss"]))
        except (ValueError, TypeError):
            cvss = c["default_cvss"]
        score += cvss * c["cvss_multiplier"]

        if str(row.get("asset_exposure", "")).strip().lower() == "internet" or \
           str(row.get("internet_exposed", "")).strip().lower() == "yes":
            score += c["internet_exposure_score"]

        if str(row.get("exploit_available", "")).strip().lower() == "yes":
            score += c["exploit_available_score"]

        if pd.notna(row.get("threat_actor")):
            score += c["threat_actor_present_score"]

        if str(row.get("ransomware_association", "")).strip().lower() == "yes":
            score += c["ransomware_association_score"]

        if str(row.get("cve", "")).upper() in self.kev_ransomware:
            score += c["cisa_kev_ransomware_score"]

        # Threat-intel exploit maturity: weaponized or active exploitation in the wild
        # scores independently of KEV so CVEs that lag behind KEV aren't under-scored
        if str(row.get("exploit_maturity", "")).strip().lower() in ("weaponized", "active exploitation"):
            score += c["exploit_maturity_score"]

        compliance = str(row.get("compliance_scope", "")).lower()
        if "pci" in compliance or "gdpr" in compliance:
            score += c["compliance_critical_score"]

        if str(row.get("criticality", "")).strip().lower() == "critical":
            score += c["criticality_score"]

        if str(row.get("patch_available", "")).strip().lower() == "no":
            score += c["patch_unavailable_score"]

        return min(score, c["max_score"])


if __name__ == "__main__":
    assets  = DataLoader.load_csv("data/assets.csv")
    vulns   = DataLoader.load_csv("data/vulnerabilities.csv")
    threat  = DataLoader.load_csv("data/threat_intelligence.csv")
    biz     = DataLoader.load_csv("data/business_services.csv")
    _       = DataLoader.load_csv("data/remediation_guidance.csv")

    kev_cves, kev_ransomware = DataLoader.fetch_cisa_kev()

    threat_agg = ThreatIntelProcessor.aggregate_by_cve(threat)
    ThreatIntelProcessor.log_unmatched(threat_agg, set(vulns["cve"].dropna()))

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
    merged["risk_score"]        = merged.apply(scorer.score_row, axis=1)
    merged["in_kev"]            = merged["cve"].str.upper().isin(kev_cves)
    merged["in_kev_ransomware"] = merged["cve"].str.upper().isin(kev_ransomware)

    _exploit    = merged.get("exploit_available",    pd.Series("", index=merged.index)).fillna("").str.strip().str.lower() == "yes"
    _ransomware = merged.get("ransomware_association", pd.Series("", index=merged.index)).fillna("").str.strip().str.lower() == "yes"
    _maturity   = merged.get("exploit_maturity",     pd.Series("", index=merged.index)).fillna("").str.strip().str.lower().isin(["weaponized", "active exploitation"])
    merged["kev_gap_flag"] = (_exploit & (_ransomware | _maturity) & (~merged["in_kev"])).astype(bool)
    gap_cves = merged[merged["kev_gap_flag"]]["cve"].drop_duplicates().tolist()
    if gap_cves:
        print(f"INFO: {len(gap_cves)} CVE(s) not in KEV but flagged for manual review: {gap_cves}", flush=True)

    top5 = (
        merged.sort_values("risk_score", ascending=False)
        .drop_duplicates(subset=["asset_id", "cve"])
        .head(5)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("  TAWASALPAY — TOP 5 CYBER RISKS")
    print("=" * 70 + "\n")

    for i, row in top5.iterrows():
        rank      = i + 1
        asset     = row.get("asset_name", row.get("asset_id", "Unknown"))
        cve       = row.get("cve", "N/A")
        vuln_name = row.get("vulnerability_name", "Unknown vulnerability")
        service   = row.get("business_service_x", row.get("business_service", "Unknown service"))
        score     = row.get("risk_score", 0)
        cvss      = row.get("cvss", "?")
        actor     = row.get("threat_actor", None)
        campaign  = row.get("campaign_name", None)
        ransomware = str(row.get("ransomware_association", "")).lower() == "yes"
        in_kev    = row.get("in_kev", False)
        exposed   = str(row.get("asset_exposure", row.get("internet_exposed", ""))).lower() in ["internet", "yes"]
        edr       = str(row.get("edr_installed", "")).strip().lower() == "yes"
        compliance = row.get("compliance_scope", "")

        reasons = []
        if exposed:
            reasons.append("internet-exposed asset")
        if actor:
            reasons.append(f"active campaign by {actor}" + (f" ({campaign})" if campaign else ""))
        if ransomware:
            reasons.append("ransomware-associated threat actor")
        if in_kev:
            reasons.append("confirmed in CISA KEV catalog")
        if compliance:
            reasons.append(f"compliance scope: {compliance}")
        if not edr:
            reasons.append("no EDR installed")
        rationale = "; ".join(reasons) if reasons else "high CVSS on critical asset"

        print(f"RISK #{rank}  (Score: {score:.0f}/100)")
        print(f"  Asset      : {asset}")
        print(f"  CVE        : {cve}  (CVSS {cvss})")
        print(f"  Vuln       : {vuln_name}")
        print(f"  Service    : {service}")
        if actor:
            print(f"  Threat actor: {actor} — {campaign}")
        print(f"  Why #{rank}      : {rationale}.")
        print()

    cols = ["asset_name", "cve", "cvss", "risk_score", "threat_actor", "campaign_name"]
    available = [c for c in cols if c in top5.columns]
    print(tabulate(top5[available], headers="keys", tablefmt="rounded_outline", showindex=False))
    print()
