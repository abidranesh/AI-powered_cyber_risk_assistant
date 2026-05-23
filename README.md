# AI-Powered Cyber Risk Assistant

A risk scoring and analysis tool that reads asset, vulnerability, and threat intelligence data from CSV files, computes a weighted risk score for each vulnerability, cross-references the live CISA KEV catalog, retrieves the most relevant NIST SP 800-53 control via semantic search, then asks an LLM to write a short analyst brief for each of the top 5 risks. There's both a CLI pipeline and a FastAPI backend with a browser dashboard.

---

## How to run it locally

**Install dependencies**

```bash
pip install -r requirements.txt
```

On first run, FastEmbed will download the `BAAI/bge-small-en-v1.5` embedding model (~50MB) and cache it locally. That's a one-time thing.

**Set your API key**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at https://console.groq.com

**Run the CLI pipeline**

```bash
python main.py
```

This loads the pre-built NIST vector store from `chroma_db/`, reads the CSVs from `data/`, fetches the CISA KEV catalog live over HTTP, computes risk scores, retrieves NIST controls via semantic search, generates LLM summaries via Groq, and writes the output to `risk_report.md`. The top 5 risks also print to the terminal in a formatted table.

**Run the API server and dashboard**

```bash
uvicorn api_server:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser. The `index.html` dashboard is served automatically. The API loads the NIST vector store in a background thread at startup, so the first request to `/api/risks` doesn't block — you'll get risk scores immediately and NIST/LLM enrichment fills in once the vector store is ready.

**Health check**

```
GET http://localhost:8000/health
```

Returns whether the risk engine, RAG pipeline, and Groq key are all present.

**Rebuilding the vector store**

The `chroma_db/` directory is committed to the repo, so you don't need to rebuild it. If you delete it, `main.py` will rebuild automatically from `data/nist_controls.csv` the next time it runs. The build batches embedding requests with a short delay between batches to stay within rate limits (this was originally written for Gemini's free tier — with FastEmbed running locally there's actually no limit, but the batching logic is still there and harmless).

---

## Project structure

```
main.py                  CLI: score → NIST retrieval → LLM → risk_report.md
api_server.py            FastAPI backend, port 8000, serves index.html + /api/risks
risk_engine.py           DataLoader, ThreatIntelProcessor, RiskScorer classes
rag_pipeline.py          Build/load Chroma vector store, retrieve NIST controls
llm_summariser.py        LangChain LCEL chain backed by Groq (llama-3.1-8b-instant)
index.html               Browser dashboard
data/
  assets.csv             Asset inventory: type, exposure, criticality, EDR status
  vulnerabilities.csv    CVEs per asset with CVSS, patch status, exploit availability
  threat_intelligence.csv  Threat actor campaigns mapped to CVE IDs
  business_services.csv  Services with compliance scope, business impact, RTO
  nist_controls.csv      NIST SP 800-53 rev5 catalog (~2,600 controls)
  remediation_guidance.csv  Finding-type-to-action lookup table
chroma_db/               Persisted ChromaDB vector store (committed; no rebuild needed)
docs/                    Sample generated reports
```

---

## What changed during development

The first version made direct API calls to Gemini — just the SDK, no abstraction layer. For something small that's fine, but as soon as I had a prompt template, an embedding call, and a vector store retrieval all happening in sequence, the glue code got messy and tightly coupled to a single provider. Pulling in LangChain cleaned that up. PromptTemplate and LCEL chaining handle the prompt construction and model call in a readable pipeline, the Chroma wrapper abstracts the vector store operations, and swapping components is now a one-line change rather than a rewrite.

That swap happened almost immediately after introducing LangChain. I was still using Gemini for both embeddings and LLM calls, and the free-tier rate limits made testing painful. The embedding step alone, indexing ~2,600 NIST controls in batches, kept hitting 429s and forcing long waits. I switched the LLM to Groq (llama-3.1-8b-instant), which has a much higher free quota and faster inference. For embeddings I switched to FastEmbed, which runs the `BAAI/bge-small-en-v1.5` model entirely in-process. No API key, no quota, no per-request cost. The NIST catalog is large enough that removing the rate limit concern from the embedding step made a real difference.

---

## Supporting questions

### What data did I embed, and what did I query as structured records?

What was embedded: Only nist_controls.csv — the NIST SP 800-53 Rev.5 catalog — goes into the vector store. Each row is dense policy prose: control statements, supplemental guidance, discussion text. You can't do an exact-match lookup on that; given a risk context like "unauthenticated RCE on an internet-exposed VPN appliance associated with a ransomware campaign," there's no column to filter on — you need semantic search to find that SI-2 or RA-5 is the right control.

What was queried as structured records: Everything else — assets, vulnerabilities, threat intel, business services — stays as CSV and gets queried with pandas merges and conditional logic. These are tabular records with well-defined foreign keys and exact-match questions: does this CVE appear in the KEV catalog? Is this asset internet-exposed? A join and a set lookup is the right tool here; embeddings would add noise with no benefit.

### Where does it go wrong?

**1. Synthetic CVE IDs silently get no KEV score.**
The vulnerability CSV uses identifiers like `CVE-SYN-2026-0001`, which don't exist in the real CISA KEV feed. The scoring logic fetches the live KEV catalog on every run and checks for matches — but those checks will always return False for synthetic IDs. The `in_kev` flag stays False and the `cisa_kev_ransomware_score` (+10 points) never fires, with no warning. In production, this would only surface correctly if the vulnerability CSV contains real CVE IDs, and there's currently no validation step that checks whether any CVEs actually matched the KEV or flags a suspiciously empty result. A simple check — log a warning if zero of N CVEs matched the live KEV — would make this failure visible rather than silent.

**2. NIST retrieval returns a result regardless of how good the match is.**
`retrieve_nist_control` always returns the top-1 vector similarity result, even when the score is low. There's no relevance threshold — if the query is vague or the vulnerability has no close analog in the NIST catalog, the function still returns a control, and the LLM still writes a confident 3–4 sentence analysis grounding its recommendations in it. A weak semantic match produces a plausible-sounding but potentially wrong remediation recommendation. The fix is straightforward: use `similarity_search_with_score` instead of `similarity_search`, and skip LLM enrichment when the best match falls below a defined distance threshold. Right now that check doesn't exist anywhere in the pipeline.

**3. Multi-actor threat intel collapses to one record per CVE.**
`ThreatIntelProcessor.aggregate_by_cve` groups threat intelligence by CVE and keeps only the single highest-confidence row per CVE. If two distinct threat actors are both actively exploiting the same vulnerability — realistic for any widely-published CVE — only one gets attached to the merged risk record. The risk score gets `threat_actor_present_score` (+15) once, but the actual threat picture is more severe. The scoring doesn't account for multiple confirmed actors, and the LLM brief names only one threat actor even when several are present. Keeping all matched actors per CVE (as a list) and adding a small score increment per additional confirmed actor would make this more accurate.

### What would I change?

The biggest gap is that everything runs on static CSV snapshots. In a real organization, data doesn't sit still — new CVEs get published daily, assets appear and disappear, threat intel updates continuously, and CISA adds new KEV entries every week or two. Right now, the only way risk scores change is if someone manually reruns the script. The system should be watching for meaningful changes and rescoring automatically: a scheduled job or event-driven trigger that detects when the KEV feed has new entries, when the vulnerability scanner drops a new export, or when a threat intel update arrives. A lightweight approach would be to hash the input CSV rows on each run and only regenerate LLM summaries for records where something actually changed, avoiding unnecessary API calls. The reporting layer could then surface a "changed since last run" indicator so an analyst immediately sees what's new rather than having to compare two reports side by side. Without that, the system is a snapshot tool rather than a monitoring tool — useful for a point-in-time report, but not something you'd trust to catch a newly active threat actor or a KEV entry that just dropped.
