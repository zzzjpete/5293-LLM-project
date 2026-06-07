# find_paper_quotes pipeline - ported verbatim from notebook cell 19.
# Uses web_search from .tools; otherwise self-contained from the original cell.
# ruff: noqa  (verbatim port; keep original style)
import io, os, re, json, math, requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from PyPDF2 import PdfReader
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .tools import web_search


# ============================================================
# FIND SOURCES + EXTRACT SUPPORTING QUOTES PIPELINE
# General-purpose, claim-aware, elite-source-aware, no topic hardcoding
# ============================================================


import os
import re
import json
import math
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage




# ============================================================
# 1. Fetch / parsing helpers
# ============================================================


def clean_html_text(html: str) -> str:
   soup = BeautifulSoup(html, "html.parser")


   for tag in soup(["script", "style", "nav", "footer", "header"]):
       tag.decompose()


   text = soup.get_text(separator=" ")
   text = re.sub(r"\s+", " ", text)
   return text.strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 20) -> str:
    """
    Extract text from PDF bytes.
    Used by fetch_webpage_text when a search result points to a PDF.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []

        for i, page in enumerate(reader.pages[:max_pages]):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {i+1} ---\n{page_text}")

        return "\n\n".join(text_parts).strip()

    except Exception as e:
        return f"Error extracting PDF text: {e}"




def normalize_fetch_url(url: str) -> str:
   """
   Generic source-normalization rule.
   Prefer full-text PDFs when a known abstract page has a direct PDF form.
   """
   url = url.strip()


   arxiv_match = re.match(r"https?://arxiv\.org/abs/([^?#]+)", url)
   if arxiv_match:
       return f"https://arxiv.org/pdf/{arxiv_match.group(1)}.pdf"


   return url




def fetch_webpage_text(url: str, max_chars: int = 60000) -> str:
   """
   Fetch readable text from an HTML page or PDF URL.
   Requires extract_text_from_pdf_bytes() to already exist.
   """
   try:
       fetch_url = normalize_fetch_url(url)


       headers = {"User-Agent": "ResearchAssistantAgent/1.0"}
       response = requests.get(fetch_url, headers=headers, timeout=25)
       response.raise_for_status()


       content_type = response.headers.get("Content-Type", "").lower()


       if "pdf" in content_type or fetch_url.lower().endswith(".pdf"):
           text = extract_text_from_pdf_bytes(response.content)
       else:
           text = clean_html_text(response.text)


       return text[:max_chars]


   except Exception as e:
       return f"Error fetching {url}: {e}"




def parse_web_search_results(search_result: str):
   """
   Parse web_search output into records.
   Expects:
   1. [Title](url)
      snippet
   """
   records = []
   blocks = re.split(r"\n\s*\n", search_result.strip())


   for block in blocks:
       link_match = re.search(r"\[(.*?)\]\((https?://[^)]+)\)", block)
       if not link_match:
           continue


       title = link_match.group(1).strip()
       url = link_match.group(2).strip()


       snippet = re.sub(
           r"^\s*\d+\.\s*\[.*?\]\(https?://[^)]+\)\s*",
           "",
           block
       ).strip()
       snippet = re.sub(r"\s+", " ", snippet)


       records.append({
           "title": title,
           "url": url,
           "snippet": snippet,
           "publication": "",
       })


   return records




# ============================================================
# 2. Cleaning + acronym + term helpers
# ============================================================


def clean_reference_request(query: str) -> str:
   q = query.strip()


   instruction_patterns = [
       r"\bfind papers to reference and quotes to support this\b",
       r"\bfind papers to reference\b",
       r"\bfind papers\b",
       r"\bfind references\b",
       r"\bfind sources\b",
       r"\bpaper to reference\b",
       r"\bpapers to reference\b",
       r"\breferences for\b",
       r"\bquotes to support this\b",
       r"\bquotes supporting this\b",
       r"\bevidence to support this\b",
       r"\bsupport this claim\b",
       r"\bsource this\b",
       r"\badd ref\b",
       r"\badd reference\b",
       r"\(add ref\)",
       r"\(add reference\)",
   ]


   for pat in instruction_patterns:
       q = re.sub(pat, "", q, flags=re.IGNORECASE)


   q = re.sub(r"\s+", " ", q).strip()
   q = q.replace("().", ".").replace("( ).", ".")
   q = re.sub(r"\s+\.", ".", q)
   q = re.sub(r"\.\s*\.$", ".", q)
   return q




def extract_parenthetical_acronyms(text: str):
   """
   Example:
   soil organic carbon (SOC) -> {"SOC": "soil organic carbon"}
   """
   pairs = re.findall(r"([A-Za-z][A-Za-z\s\-]+?)\s*\(([A-Z]{2,})\)", text)
   acronym_map = {}


   for phrase, acronym in pairs:
       phrase = re.sub(r"\s+", " ", phrase.strip())
       if len(phrase.split()) >= 2:
           acronym_map[acronym] = phrase


   return acronym_map




def expand_defined_acronyms(text: str, acronym_map: dict) -> str:
   expanded = text


   for acronym, phrase in acronym_map.items():
       expanded = re.sub(
           rf"\b{re.escape(acronym)}\b",
           f"{phrase} {acronym}",
           expanded
       )


   return expanded




def extract_candidate_method_names(text: str):
   """
   General method/model name extractor.
   Finds names like XGBoost, LightGBM, BERT, LoRA, RAG, SVM.
   """
   names = re.findall(
       r"\b(?:[A-Z]{2,}[A-Za-z0-9\-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9\-]*)\b",
       text
   )


   cleaned = []
   seen = set()


   for n in names:
       key = n.lower()
       if key not in seen:
           cleaned.append(n)
           seen.add(key)


   return cleaned




def tokenize_for_matching(text: str):
   words = re.findall(r"[a-zA-Z0-9]+", text.lower())


   stop = {
       "the", "and", "for", "with", "this", "that", "from", "into",
       "using", "used", "use", "uses", "model", "models", "method",
       "methods", "paper", "article", "journal", "study", "claim",
       "support", "reference", "references", "source", "sources",
       "based", "prediction", "predicting", "predict", "analysis",
       "data", "result", "results", "show", "shows", "shown",
       "find", "quotes", "evidence", "approach", "framework",
       "well", "suited", "due", "ability", "across", "specified",
       "while", "through", "their", "its", "because", "therefore",
       "also", "such", "these", "those", "being", "have", "has",
       "had", "can", "may", "might"
   }


   return [w for w in words if len(w) > 2 and w not in stop]




def keyword_overlap_score(text: str, keywords) -> float:
   if not keywords:
       return 0.0


   text_low = text.lower()
   hits = 0


   for kw in keywords:
       if kw.lower() in text_low:
           hits += 1


   return hits / len(keywords)




def extract_informative_phrases(text: str, max_phrases: int = 8):
   """
   Generic phrase extractor.
   Pulls multi-word phrases that are likely domain/application concepts.
   No topic-specific patterns.
   """
   text = re.sub(r"[^A-Za-z0-9\s\-]", " ", text)
   text = re.sub(r"\s+", " ", text).strip()


   words = text.split()
   stop = {
       "the", "and", "for", "with", "this", "that", "from", "into",
       "using", "used", "uses", "model", "models", "method", "methods",
       "paper", "study", "claim", "support", "well", "suited", "due",
       "ability", "across", "while", "through", "its", "their", "specified",
       "can", "may", "are", "is", "was", "were", "has", "have", "had",
       "to", "of", "in", "on", "by", "as", "an", "a"
   }


   phrases = []
   for n in [4, 3, 2]:
       for i in range(len(words) - n + 1):
           gram = words[i:i+n]
           if any(w.lower() in stop for w in gram):
               continue
           phrase = " ".join(gram)
           if len(phrase) >= 8:
               phrases.append(phrase.lower())


   # Keep order, remove near duplicates
   out = []
   seen = set()
   for p in phrases:
       if p not in seen:
           out.append(p)
           seen.add(p)
       if len(out) >= max_phrases:
           break


   return out




def generic_required_terms_for_claim(claim_text: str, method: str = "", need: str = "concept"):
   """
   Builds required terms without hardcoding a domain.
   For application claims, prefer method + multi-word domain phrases.
   """
   terms = []


   if method:
       terms.append(method)


   phrases = extract_informative_phrases(claim_text, max_phrases=8)


   if need == "application":
       # Application claims need domain preservation, so phrases matter more than single words.
       terms.extend(phrases[:5])
   else:
       terms.extend(tokenize_for_matching(claim_text)[:6])


   return list(dict.fromkeys([t for t in terms if t]))




def generic_quote_keywords_for_claim(claim_text: str, method: str = ""):
   keywords = []
   if method:
       keywords.append(method)


   keywords.extend(extract_informative_phrases(claim_text, max_phrases=8))
   keywords.extend(tokenize_for_matching(claim_text)[:12])


   return list(dict.fromkeys([k for k in keywords if k]))




def infer_need(sentence: str) -> str:
   low = sentence.lower()


   application_cues = [
       "prediction", "predict", "well suited", "applied", "used for",
       "application", "forecast", "classification", "detection",
       "estimation", "mapping", "diagnosis", "recommendation"
   ]


   method_cues = [
       "method", "algorithm", "loss function", "regularization",
       "objective", "decision tree", "ensemble", "architecture",
       "optimizer", "training", "minimize", "fits", "learns"
   ]


   if any(x in low for x in application_cues):
       return "application"


   if any(x in low for x in method_cues):
       return "method"


   return "concept"




def resolve_referential_method(sentence: str, explicit_method: str, previous_method: str):
   """
   Generic reference resolution:
   If current claim says "the model", "this method", etc.,
   inherit previous explicit method/model.
   """
   if explicit_method:
       return explicit_method


   low = sentence.lower()
   referential_patterns = [
       r"\bthe model\b",
       r"\bthis model\b",
       r"\bthe method\b",
       r"\bthis method\b",
       r"\bthe algorithm\b",
       r"\bthis algorithm\b",
       r"\bit\b",
       r"\bits\b",
   ]


   if previous_method and any(re.search(p, low) for p in referential_patterns):
       return previous_method


   return ""




# ============================================================
# 3. Claim planner
# ============================================================


def split_claim_sentences(cleaned_text: str):
   """
   Deterministically split cleaned paragraph into citation claims.
   One sentence = one claim.
   """
   text = cleaned_text.strip()
   parts = re.split(r"(?<=[.!?])\s+", text)


   sentences = []
   for s in parts:
       s = re.sub(r"\s+", " ", s.strip())
       s = re.sub(r"\s+\.$", ".", s)
       if len(s.split()) >= 5:
           sentences.append(s)


   return sentences if sentences else [text]




def fallback_claim_from_sentence(sentence: str, claim_id: str, acronym_map: dict, previous_method: str = ""):
   """
   Build a safe default claim object from one sentence.
   """
   sentence = expand_defined_acronyms(sentence, acronym_map)
   names = extract_candidate_method_names(sentence)


   explicit_method = names[0] if names else ""
   method = resolve_referential_method(sentence, explicit_method, previous_method)


   need = infer_need(sentence)
   terms = tokenize_for_matching(sentence)
   phrases = extract_informative_phrases(sentence)


   if need == "application":
       domain = " ".join(phrases[:2]) if phrases else ""
   else:
       domain = ""


   required_terms = generic_required_terms_for_claim(sentence, method=method, need=need)
   quote_keywords = generic_quote_keywords_for_claim(sentence, method=method)


   search_query = build_generic_search_query(
       claim_text=sentence,
       need=need,
       method=method,
       domain=domain,
       terms=terms,
       phrases=phrases
   )


   return {
       "claim_id": claim_id,
       "claim": sentence,
       "need": need,
       "method_or_model": method,
       "domain": domain,
       "what_to_support": sentence,
       "search_query": search_query,
       "required_terms": required_terms,
       "quote_keywords": quote_keywords,
   }




def build_generic_search_query(claim_text: str, need: str, method: str = "", domain: str = "", terms=None, phrases=None):
   """
   Generic claim-aware search query builder.
   Avoids hardcoded topics.
   """
   terms = terms or tokenize_for_matching(claim_text)
   phrases = phrases or extract_informative_phrases(claim_text)


   parts = []


   if method:
       parts.append(method)


   if need == "method":
       parts.extend(["original paper", "method", "objective", "loss", "regularization"])
       parts.extend(terms[:5])


   elif need == "application":
       # Application search must preserve the actual domain phrases.
       parts.extend(phrases[:4])
       parts.extend(terms[:8])
       parts.extend(["paper", "machine learning"])


   else:
       parts.extend(phrases[:3])
       parts.extend(terms[:8])
       parts.extend(["source", "paper", "article"])


   # De-duplicate while preserving order.
   cleaned = []
   seen = set()
   for p in parts:
       p = str(p).strip()
       key = p.lower()
       if p and key not in seen:
           cleaned.append(p)
           seen.add(key)


   return " ".join(cleaned[:16])




def repair_llm_claim_metadata(base: dict, llm_claim: dict, acronym_map: dict, previous_method: str = ""):
   """
   Generic repair layer after LLM planning.
   Keeps the LLM useful, but prevents generic application queries.
   """
   claim_text = base["claim"]
   need = llm_claim.get("need", base["need"])


   if need not in {"method", "application", "concept"}:
       need = base["need"]


   explicit_method = llm_claim.get("method_or_model", base.get("method_or_model", ""))
   method = resolve_referential_method(claim_text, explicit_method, previous_method)


   phrases = extract_informative_phrases(claim_text)
   terms = tokenize_for_matching(claim_text)


   domain = llm_claim.get("domain", base.get("domain", ""))
   if need == "application" and (not domain or len(domain.split()) <= 1):
       domain = " ".join(phrases[:2]) if phrases else domain


   required_terms = llm_claim.get("required_terms", base["required_terms"])
   quote_keywords = llm_claim.get("quote_keywords", base["quote_keywords"])


   if not isinstance(required_terms, list):
       required_terms = base["required_terms"]


   if not isinstance(quote_keywords, list):
       quote_keywords = base["quote_keywords"]


   # Remove generic filler from required terms.
   filler = {
       "well", "suited", "due", "ability", "its", "their", "this",
       "that", "model", "method", "paper", "study", "result", "results"
   }


   required_terms = [
       t for t in required_terms
       if isinstance(t, str) and t.strip() and t.lower().strip() not in filler
   ]


   # For application claims, force terms to preserve actual content.
   if need == "application":
       required_terms = generic_required_terms_for_claim(claim_text, method=method, need=need)
       quote_keywords = generic_quote_keywords_for_claim(claim_text, method=method)


   # For method claims, keep method strongly represented.
   if need == "method" and method:
       required_terms = [method] + [t for t in required_terms if t.lower() != method.lower()]
       quote_keywords = [method] + [t for t in quote_keywords if t.lower() != method.lower()]


   llm_query = llm_claim.get("search_query", base["search_query"])
   generic_query = is_generic_search_query(llm_query, claim_text)


   if generic_query or need == "application":
       search_query = build_generic_search_query(
           claim_text=claim_text,
           need=need,
           method=method,
           domain=domain,
           terms=terms,
           phrases=phrases
       )
   else:
       search_query = expand_defined_acronyms(llm_query, acronym_map)


   return {
       "claim_id": base["claim_id"],
       "claim": claim_text,
       "need": need,
       "method_or_model": method,
       "domain": domain,
       "what_to_support": expand_defined_acronyms(
           llm_claim.get("what_to_support", claim_text),
           acronym_map
       ),
       "search_query": search_query,
       "required_terms": list(dict.fromkeys(required_terms)),
       "quote_keywords": list(dict.fromkeys(quote_keywords)),
   }




def is_generic_search_query(search_query: str, claim_text: str) -> bool:
   """
   Detects whether a query has lost the actual claim content.
   General rule: if query shares little with claim's informative terms,
   it is too generic.
   """
   q_terms = set(tokenize_for_matching(search_query))
   c_terms = set(tokenize_for_matching(claim_text))


   if not c_terms:
       return False


   overlap = len(q_terms & c_terms) / max(1, len(c_terms))


   generic_words = {
       "paper", "results", "study", "research", "application",
       "environmental", "science", "predictive", "modeling",
       "machine", "learning"
   }


   mostly_generic = len(q_terms - generic_words) <= 2


   return overlap < 0.25 or mostly_generic




def llm_plan_citation_claims(user_query: str):
   """
   One sentence = one claim.
   The LLM may enrich metadata, but it cannot create extra claims.
   Generic repair layer prevents topic loss.
   """
   acronym_map = extract_parenthetical_acronyms(user_query)
   fallback_cleaned = clean_reference_request(user_query)
   fallback_expanded = expand_defined_acronyms(fallback_cleaned, acronym_map)


   sentence_claims = []
   previous_method = ""


   for i, sentence in enumerate(split_claim_sentences(fallback_expanded)):
       base = fallback_claim_from_sentence(
           sentence,
           f"C{i+1}",
           acronym_map,
           previous_method=previous_method
       )
       if base.get("method_or_model"):
           previous_method = base["method_or_model"]
       sentence_claims.append(base)


   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


   system = """
You are helping build a research citation assistant.


You will receive a fixed list of sentence-level claims.


Critical rules:
- Return the same number of claims you receive.
- Preserve the same claim_id values.
- Do not split claims.
- Do not merge claims.
- Do not add new claims.
- Do not remove claims.
- You may only enrich metadata fields.
- If a claim says "the model", "this model", "the method", "this method", or "it",
 resolve it to the most recent explicit method/model from previous claims.
- For application claims, the search_query must preserve both:
 1. the method/model if known
 2. the concrete application domain from the claim
- Do not use vague words like "well", "suited", "due", "ability", "its", or "model"
 as required_terms.


For each existing claim, fill in:
- need: method, application, or concept
- method_or_model
- domain
- what_to_support
- search_query
- required_terms
- quote_keywords


Return only valid JSON.


JSON format:
{
 "claims": [
   {
     "claim_id": "C1",
     "claim": "...",
     "need": "method | application | concept",
     "method_or_model": "...",
     "domain": "...",
     "what_to_support": "...",
     "search_query": "...",
     "required_terms": ["..."],
     "quote_keywords": ["..."]
   }
 ]
}
"""


   human = f"""
Detected acronym definitions:
{json.dumps(acronym_map, indent=2)}


Fixed sentence-level claims:
{json.dumps(sentence_claims, indent=2)}


Enrich these claims without changing the number of claims.
"""


   try:
       response = llm.invoke([
           SystemMessage(content=system),
           HumanMessage(content=human),
       ])


       text = response.content.strip()


       if "```json" in text:
           text = text.split("```json")[1].split("```")[0].strip()
       elif "```" in text:
           text = text.split("```")[1].split("```")[0].strip()


       parsed = json.loads(text)


       if "claims" not in parsed:
           raise ValueError("Missing claims key.")


       enriched = parsed["claims"]


       if len(enriched) != len(sentence_claims):
           raise ValueError(
               f"LLM returned {len(enriched)} claims, expected {len(sentence_claims)}."
           )


       final_claims = []
       previous_method = ""


       for base, llm_claim in zip(sentence_claims, enriched):
           repaired = repair_llm_claim_metadata(
               base=base,
               llm_claim=llm_claim,
               acronym_map=acronym_map,
               previous_method=previous_method
           )


           if repaired.get("method_or_model"):
               previous_method = repaired["method_or_model"]


           final_claims.append(repaired)


       return {
           "cleaned_paragraph": fallback_expanded,
           "claims": final_claims,
       }


   except Exception as e:
       return {
           "cleaned_paragraph": fallback_expanded,
           "claims": sentence_claims,
           "planner_error": str(e),
       }




# ============================================================
# 4. Elite-tier source scoring
# ============================================================


def source_tier_score(url: str) -> float:
   """
   General source tiers.
   This rewards source class, not topic content.
   """
   domain = urlparse(url).netloc.lower().replace("www.", "")
   path = url.lower()


   elite_domains = [
       "nature.com",
       "science.org",
       "cell.com",
       "nejm.org",
       "thelancet.com",
       "pnas.org",
       "jmlr.org",
       "dl.acm.org",
       "acm.org",
       "ieeexplore.ieee.org",
       "ieee.org",
       "neurips.cc",
       "proceedings.mlr.press",
       "aaai.org",
       "ijcai.org",
       "kdd.org",
       "usenix.org",
       "arxiv.org",
   ]


   strong_scholarly_domains = [
       "openreview.net",
       "springer.com",
       "link.springer.com",
       "sciencedirect.com",
       "wiley.com",
       "tandfonline.com",
       "frontiersin.org",
       "plos.org",
       "mdpi.com",
       "pmc.ncbi.nlm.nih.gov",
       "pubmed.ncbi.nlm.nih.gov",
       "semanticscholar.org",
   ]


   official_or_institutional = [
       "readthedocs.io",
       "github.io",
       "edu",
       "gov",
       "usda.gov",
       "nasa.gov",
       "noaa.gov",
       "usgs.gov",
   ]


   weak_domains = [
       "medium.com",
       "towardsdatascience.com",
       "geeksforgeeks.org",
       "analyticsvidhya.com",
       "tutorialspoint.com",
       "ibm.com",
       "aws.amazon.com",
       "microsoft.com",
       "nvidia.com",
       "academia.edu",
       "researchgate.net",
       "slideshare.net",
   ]


   if any(d in domain for d in elite_domains):
       return 3.0


   if any(d in domain for d in strong_scholarly_domains):
       return 2.0


   if any(d in domain for d in official_or_institutional) or domain.endswith(".edu") or domain.endswith(".gov"):
       return 1.3


   if any(d in domain for d in weak_domains):
       return -1.5


   if path.endswith(".pdf"):
       return 0.8


   return 0.0




def paper_like_score(source_text: str) -> float:
   if not source_text:
       return 0.0


   text = source_text.lower()


   signals = [
       "abstract",
       "introduction",
       "method",
       "methods",
       "results",
       "discussion",
       "conclusion",
       "references",
       "doi",
       "proceedings",
       "journal",
       "citation",
   ]


   hits = sum(1 for s in signals if s in text)
   return min(hits / 8, 1.0)




def title_exact_method_bonus(candidate: dict) -> float:
   """
   General canonical-method bonus.
   Rewards titles that look like canonical sources for a method.
   """
   if candidate.get("need") != "method":
       return 0.0


   method = candidate.get("method_or_model", "").strip().lower()
   title = candidate.get("title", "").lower()
   url = candidate.get("url", "").lower()


   if not method:
       return 0.0


   bonus = 0.0


   if title == method:
       bonus += 250.0


   if title.startswith(method + ":") or title.startswith(method + " "):
       bonus += 220.0


   if method in title:
       bonus += 100.0


   canonical_words = [
       "system",
       "algorithm",
       "method",
       "model",
       "framework",
       "architecture",
       "approach",
       "paper",
   ]


   if method in title and any(w in title for w in canonical_words):
       bonus += 80.0


   if ".pdf" in url:
       bonus += 40.0


   return bonus




def source_quality_score(url: str, source_text: str = "") -> float:
   return source_tier_score(url) + paper_like_score(source_text)




def term_present(term: str, text: str) -> bool:
   return term.lower().strip() in text.lower()




def required_term_score(required_terms, candidate_text: str) -> float:
   if not required_terms:
       return 1.0


   hits = 0
   for term in required_terms:
       if term_present(term, candidate_text):
           hits += 1


   return hits / len(required_terms)




def title_query_match_score(title: str, query: str) -> float:
   title_terms = set(tokenize_for_matching(title))
   query_terms = tokenize_for_matching(query)


   if not query_terms:
       return 0.0


   hits = sum(1 for w in query_terms if w in title_terms)
   return hits / len(query_terms)




def snippet_query_match_score(snippet: str, query: str) -> float:
   snippet_terms = set(tokenize_for_matching(snippet))
   query_terms = tokenize_for_matching(query)


   if not query_terms:
       return 0.0


   hits = sum(1 for w in query_terms if w in snippet_terms)
   return hits / len(query_terms)




def rank_web_candidate(candidate: dict) -> float:
   quality = candidate.get("quality_score", 0.0)
   title_match = candidate.get("title_match", 0.0)
   snippet_match = candidate.get("snippet_match", 0.0)
   required_score = candidate.get("required_term_score", 1.0)
   canonical_bonus = title_exact_method_bonus(candidate)


   quality_component = 150.0 * quality
   title_component = 100.0 * title_match
   snippet_component = 75.0 * snippet_match


   base = quality_component + title_component + snippet_component + canonical_bonus


   if candidate.get("need") == "application":
       return base * (0.25 + 0.75 * required_score)


   return base * (0.50 + 0.50 * required_score)




def candidate_is_relevant(candidate: dict) -> bool:
   """
   Generic relevance gate.
   For application claims, require at least some overlap with concrete required terms.
   For method claims, allow canonical sources through.
   """
   required_terms = candidate.get("required_terms", [])
   combined = " ".join([
       candidate.get("title", ""),
       candidate.get("snippet", ""),
       candidate.get("publication", "")
   ])


   if candidate.get("need") == "application" and required_terms:
       score = required_term_score(required_terms, combined)


       # Allow partial overlap because snippets are short.
       return score > 0


   return True




# ============================================================
# 5. Claim-aware query construction + source collection
# ============================================================


def build_support_query(claim: dict) -> str:
   need = claim.get("need", "concept")
   method = claim.get("method_or_model", "").strip()
   domain = claim.get("domain", "").strip()
   support = claim.get("what_to_support", claim.get("claim", "")).strip()
   query = claim.get("search_query", support).strip()


   # Always rebuild if query is too generic.
   if is_generic_search_query(query, support):
       query = build_generic_search_query(
           claim_text=support,
           need=need,
           method=method,
           domain=domain,
           terms=tokenize_for_matching(support),
           phrases=extract_informative_phrases(support)
       )


   if need == "method":
       if method:
           return f'{method} original paper method objective loss regularization'
       return f"{query} original paper method explanation"


   if need == "application":
       return query


   return f"{query} source paper article"




def build_fallback_support_queries(claim: dict):
   """
   Generic fallback queries if a claim produces no quotes.
   No topic hardcoding.
   """
   claim_text = claim.get("claim", "")
   method = claim.get("method_or_model", "").strip()
   need = claim.get("need", "concept")
   phrases = extract_informative_phrases(claim_text, max_phrases=8)
   terms = tokenize_for_matching(claim_text)


   queries = []


   if need == "application":
       if method and phrases:
           queries.append(" ".join([method] + phrases[:3] + ["paper"]))
           queries.append(" ".join([method] + terms[:8] + ["empirical study"]))
       if phrases:
           queries.append(" ".join(phrases[:4] + ["machine learning paper"]))
       queries.append(" ".join(terms[:10] + ["research article"]))


   elif need == "method":
       if method:
           queries.append(f"{method} original paper")
           queries.append(f"{method} method objective function")
       queries.append(" ".join(terms[:10] + ["method paper"]))


   else:
       queries.append(" ".join(terms[:10] + ["source paper"]))
       if phrases:
           queries.append(" ".join(phrases[:4] + ["review article"]))


   return list(dict.fromkeys([q for q in queries if q.strip()]))




def enrich_candidate_record(r: dict, claim: dict, support_query: str):
   combined = " ".join([r["title"], r["snippet"], r.get("publication", "")])


   r["claim_id"] = claim.get("claim_id", "")
   r["claim"] = claim.get("claim", "")
   r["need"] = claim.get("need", "")
   r["method_or_model"] = claim.get("method_or_model", "")
   r["domain"] = claim.get("domain", "")
   r["what_to_support"] = claim.get("what_to_support", claim.get("claim", ""))
   r["query_used"] = support_query
   r["required_terms"] = claim.get("required_terms", [])
   r["quote_keywords"] = claim.get("quote_keywords", [])
   r["quality_score"] = source_quality_score(r["url"], "")
   r["title_match"] = title_query_match_score(r["title"], support_query)
   r["snippet_match"] = snippet_query_match_score(r["snippet"], support_query)
   r["required_term_score"] = required_term_score(r["required_terms"], combined)
   r["passes_relevance_gate"] = candidate_is_relevant(r)
   r["canonical_bonus"] = title_exact_method_bonus(r)
   r["rank_score"] = rank_web_candidate(r)


   return r




def collect_web_candidates(user_query: str, num_per_query: int = 8, verbose: bool = True):
   plan = llm_plan_citation_claims(user_query)
   all_candidates = []


   if verbose:
       print("=" * 100)
       print("STEP 1: CLEANED PARAGRAPH")
       print("=" * 100)
       print(plan["cleaned_paragraph"])


       print("\n" + "=" * 100)
       print("STEP 2: ATOMIC CITATION CLAIMS + WEB SEARCH QUERIES")
       print("=" * 100)


   for claim in plan["claims"]:
       claim_id = claim.get("claim_id", f"C{len(all_candidates)+1}")
       support_query = build_support_query(claim)


       if verbose:
           print(f"\nClaim ID: {claim_id}")
           print(f"Need type: {claim.get('need', 'concept')}")
           print(f"Method/model: {claim.get('method_or_model', '')}")
           print(f"Domain: {claim.get('domain', '')}")
           print(f"Claim: {claim.get('claim', '')}")
           print(f"Supports: {claim.get('what_to_support', claim.get('claim', ''))}")
           print(f"Web query: {support_query}")
           print(f"Required terms: {claim.get('required_terms', [])}")
           print(f"Quote keywords: {claim.get('quote_keywords', [])}")


       search_result = web_search.invoke(support_query)
       records = parse_web_search_results(search_result)[:num_per_query]


       for r in records:
           if not r.get("url"):
               continue


           r = enrich_candidate_record(r, claim, support_query)


           if not r["passes_relevance_gate"]:
               continue


           all_candidates.append(r)


   deduped = {}
   for c in all_candidates:
       key = (c["url"], c["claim_id"])
       if key not in deduped or c["rank_score"] > deduped[key]["rank_score"]:
           deduped[key] = c


   ranked = sorted(deduped.values(), key=lambda x: x["rank_score"], reverse=True)


   if verbose:
       print("\n" + "=" * 100)
       print("STEP 3: RANKED WEB CANDIDATES AFTER ELITE SOURCE SCORING")
       print("=" * 100)


       for i, c in enumerate(ranked[:20], 1):
           print(f"\n[{i}] {c['title']}")
           print(f"Claim ID: {c['claim_id']}")
           print(f"Need: {c['need']}")
           print(f"Method/model: {c.get('method_or_model', '')}")
           print(f"Domain: {c.get('domain', '')}")
           print(f"Source tier score: {source_tier_score(c['url']):.2f}")
           print(f"Quality score: {c['quality_score']:.2f}")
           print(f"Title match: {c['title_match']:.3f}")
           print(f"Snippet match: {c['snippet_match']:.3f}")
           print(f"Required term score: {c['required_term_score']:.3f}")
           print(f"Canonical bonus: {c['canonical_bonus']:.1f}")
           print(f"Rank score: {c['rank_score']:.1f}")
           print(f"URL: {c['url']}")
           print(f"Snippet: {c['snippet']}")


   return plan, ranked




def collect_additional_candidates_for_claim(claim: dict, num_per_query: int = 8):
   """
   Generic fallback candidate collection.
   """
   all_candidates = []


   for support_query in build_fallback_support_queries(claim):
       search_result = web_search.invoke(support_query)
       records = parse_web_search_results(search_result)[:num_per_query]


       for r in records:
           if not r.get("url"):
               continue


           r = enrich_candidate_record(r, claim, support_query)


           if r["passes_relevance_gate"]:
               all_candidates.append(r)


   deduped = {}
   for c in all_candidates:
       key = (c["url"], c["claim_id"])
       if key not in deduped or c["rank_score"] > deduped[key]["rank_score"]:
           deduped[key] = c


   return sorted(deduped.values(), key=lambda x: x["rank_score"], reverse=True)








# ============================================================
# 6. Quote mining helpers
# One best quote per claim from best usable source
# No topic hardcoding
# ============================================================


def normalize_quote_text(text: str) -> str:
   """
   Normalize extracted PDF/HTML text while preserving quote meaning.
   """
   if text is None:
       return ""


   text = str(text)


   # PDF cleanup
   text = re.sub(r"-\s*\n\s*", "", text)
   text = re.sub(r"\s*\n\s*", " ", text)
   text = re.sub(r"\s+", " ", text)


   # Punctuation cleanup
   text = re.sub(r"\s+([,.;:!?])", r"\1", text)
   text = re.sub(r"([(\[{])\s+", r"\1", text)
   text = re.sub(r"\s+([)\]}])", r"\1", text)


   return text.strip()




def split_source_into_sentences(text: str):
   """
   Fast sentence splitter with character offsets.
   """
   text = normalize_quote_text(text)


   raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text)


   sentences = []
   cursor = 0


   for s in raw:
       s = normalize_quote_text(s)
       if not s:
           continue


       start = text.find(s, cursor)
       if start == -1:
           start = text.find(s)


       end = start + len(s) if start != -1 else -1
       cursor = max(cursor, end)


       sentences.append({
           "sentence": s,
           "start_char": start,
           "end_char": end,
       })


   return sentences




def quote_quality_filter(sentence: str) -> bool:
   """
   Generic source-format filter. Removes metadata, headers, reference fragments,
   and obvious non-quote text.
   """
   s = normalize_quote_text(sentence)
   low = s.lower()
   words = s.split()


   if len(words) < 8:
       return False


   # Allow multi-sentence quote windows.
   if len(words) > 160:
       return False


   bad_fragments = [
       "skip to main content",
       "advertisement",
       "cite this article",
       "download pdf",
       "rights and permissions",
       "publisher's note",
       "cookie",
       "privacy policy",
       "copyright notice",
       "author information",
       "ethics declarations",
       "data availability",
       "supplementary information",
       "references",
       "acknowledgements",
       "acknowledgments",
       "submitted on",
       "view a pdf",
       "all rights reserved",
       "google scholar",
       "crossref",
       "pubmed",
       "pmcid",
       "keywords:",
       "article info",
       "received:",
       "accepted:",
       "published:",
       "correspondence:",
       "edited by:",
       "reviewed by:",
   ]


   if any(b in low for b in bad_fragments):
       return False


   citationish = [
       "proceedings of",
       "conference on",
       "journal of",
       "volume",
       "pages",
       "doi",
       "arxiv",
       "in proceedings",
       "article google scholar",
   ]


   if sum(p in low for p in citationish) >= 2:
       return False


   # Section heading.
   if len(words) <= 14 and re.match(r"^\d+(\.\d+)*\s+[A-Z]", s):
       return False


   # Title/nav fragment.
   if "|" in s and len(words) < 35:
       return False


   # Mostly capitalized = likely title/header/author line.
   alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
   if len(alpha_words) >= 10:
       cap_ratio = sum(1 for w in alpha_words if w[:1].isupper()) / len(alpha_words)
       if cap_ratio > 0.68:
           return False


   return True




def claim_content_terms(claim: str, method: str = ""):
   """
   Terms used for matching. Comes only from the claim and resolved method.
   """
   terms = tokenize_for_matching(claim)
   phrases = extract_informative_phrases(claim, max_phrases=12)


   out = []
   if method:
       out.append(method.lower())


   out.extend([p.lower() for p in phrases])
   out.extend([t.lower() for t in terms])


   return list(dict.fromkeys([x for x in out if x]))




def term_hit_count(text: str, terms) -> int:
   low = normalize_quote_text(text).lower()
   return sum(
       1 for t in terms
       if isinstance(t, str) and t.strip() and t.lower() in low
   )




def term_hit_fraction(text: str, terms) -> float:
   if not terms:
       return 0.0
   return term_hit_count(text, terms) / len(terms)




def quote_word_coverage(sentence: str, claim: str) -> float:
   terms = list(dict.fromkeys(tokenize_for_matching(claim)))
   if not terms:
       return 0.0


   low = normalize_quote_text(sentence).lower()
   hits = sum(1 for t in terms if t in low)
   return hits / len(terms)




def quote_phrase_coverage(sentence: str, claim: str) -> float:
   phrases = extract_informative_phrases(claim, max_phrases=12)
   if not phrases:
       return 0.0


   low = normalize_quote_text(sentence).lower()
   hits = sum(1 for p in phrases if p.lower() in low)
   return hits / len(phrases)




def quote_required_coverage(sentence: str, required_terms) -> float:
   if not required_terms:
       return 0.0


   low = normalize_quote_text(sentence).lower()
   hits = sum(
       1 for t in required_terms
       if isinstance(t, str) and t.strip() and t.lower() in low
   )
   return hits / len(required_terms)




def quote_keyword_coverage(sentence: str, quote_keywords) -> float:
   if not quote_keywords:
       return 0.0


   low = normalize_quote_text(sentence).lower()
   hits = sum(
       1 for t in quote_keywords
       if isinstance(t, str) and t.strip() and t.lower() in low
   )
   return hits / len(quote_keywords)




def quote_method_bonus(sentence: str, source: dict) -> float:
   method = source.get("method_or_model", "")
   if method and method.lower() in normalize_quote_text(sentence).lower():
       return 1.0
   return 0.0




def quote_structure_bonus(sentence: str) -> float:
   """
   Generic structure bonus. Rewards formal/explanatory quotes.
   Not tied to any specific topic.
   """
   s = normalize_quote_text(sentence)
   low = s.lower()


   bonus = 0.0


   # Formal/equation-like structure.
   if any(sym in s for sym in ["=", "∑", "Ω", "λ", "γ", "‖", "||", "+"]):
       bonus += 0.7


   # Explanation structure.
   markers = [
       "where",
       "here",
       "term",
       "measures",
       "penalizes",
       "penalty",
       "complexity",
       "objective",
       "loss function",
       "helps",
       "allows",
       "enables",
       "makes",
       "effective",
       "ability",
       "due to",
       "because",
       "depends upon",
       "account for",
   ]


   if any(x in low for x in markers):
       bonus += 0.8


   return min(bonus, 1.5)




def quote_noise_penalty(sentence: str) -> float:
   """
   Generic penalty for boilerplate, title/header text, or weak fragments.
   """
   s = normalize_quote_text(sentence)
   low = s.lower()
   words = s.split()


   penalty = 0.0


   bad_terms = [
       "in this section",
       "we evaluate",
       "table",
       "figure",
       "appendix",
       "supplementary",
       "download",
       "published",
       "copyright",
       "submitted on",
       "view pdf",
       "authors",
       "keywords",
       "abstract:",
       "introduction:",
   ]


   penalty += sum(0.6 for t in bad_terms if t in low)


   if len(words) < 18 and any(t in low for t in ["abstract", "keywords", "introduction", "original research"]):
       penalty += 1.5


   alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
   if len(alpha_words) >= 10:
       cap_ratio = sum(1 for w in alpha_words if w[:1].isupper()) / len(alpha_words)
       if cap_ratio > 0.60:
           penalty += 1.0


   comma_count = s.count(",")
   if comma_count >= 5 and len(words) < 35:
       penalty += 1.2


   return penalty




def quote_claim_score(sentence: str, source: dict) -> float:
   """
   Generic claim-aware quote score.
   """
   sentence = normalize_quote_text(sentence)
   claim = source.get("claim", "")
   required_terms = source.get("required_terms", [])
   quote_keywords = source.get("quote_keywords", [])
   need = source.get("need", "concept")


   word_cov = quote_word_coverage(sentence, claim)
   phrase_cov = quote_phrase_coverage(sentence, claim)
   required_cov = quote_required_coverage(sentence, required_terms)
   keyword_cov = quote_keyword_coverage(sentence, quote_keywords)
   method_bonus = quote_method_bonus(sentence, source)
   structure_bonus = quote_structure_bonus(sentence)
   noise = quote_noise_penalty(sentence)


   length = max(1, len(sentence.split()))
   density = min(1.0, (term_hit_count(sentence, tokenize_for_matching(claim)) / length) * 8.0)


   if need == "method":
       return (
           2.1 * word_cov
           + 2.4 * phrase_cov
           + 1.6 * required_cov
           + 1.1 * keyword_cov
           + 0.9 * method_bonus
           + 1.2 * structure_bonus
           + 0.7 * density
           - 1.2 * noise
       )


   if need == "application":
       return (
           2.0 * word_cov
           + 2.8 * phrase_cov
           + 1.9 * required_cov
           + 1.2 * keyword_cov
           + 0.5 * method_bonus
           + 1.0 * structure_bonus
           + 0.7 * density
           - 1.2 * noise
       )


   return (
       2.0 * word_cov
       + 2.4 * phrase_cov
       + 1.5 * required_cov
       + 1.2 * keyword_cov
       + 0.5 * method_bonus
       + 0.9 * structure_bonus
       + 0.7 * density
       - 1.2 * noise
   )




def quote_global_score(q: dict, source: dict) -> float:
   sentence = normalize_quote_text(q["sentence"])
   claim = source.get("claim", "")
   required_terms = source.get("required_terms", [])
   quote_keywords = source.get("quote_keywords", [])


   word_score = quote_word_coverage(sentence, claim)
   phrase_score = quote_phrase_coverage(sentence, claim)
   required_score = quote_required_coverage(sentence, required_terms)
   keyword_score = quote_keyword_coverage(sentence, quote_keywords)
   method_bonus = quote_method_bonus(sentence, source)
   structure_bonus = quote_structure_bonus(sentence)
   noise = quote_noise_penalty(sentence)


   length = max(1, len(sentence.split()))
   density = min(1.0, (term_hit_count(sentence, tokenize_for_matching(claim)) / length) * 8.0)


   q["claim_term_score"] = word_score
   q["phrase_coverage_score"] = phrase_score
   q["required_term_score_quote"] = required_score
   q["quote_keyword_score"] = keyword_score
   q["method_bonus"] = method_bonus
   q["structure_bonus"] = structure_bonus
   q["quote_noise_penalty"] = noise
   q["density_score"] = density


   return (
       q.get("rerank_score", q.get("score", 0))
       + 2.0 * word_score
       + 2.6 * phrase_score
       + 1.6 * required_score
       + 1.1 * keyword_score
       + 0.7 * method_bonus
       + 1.2 * structure_bonus
       + 0.7 * density
       + 0.004 * source.get("rank_score", 0)
       + 0.30 * q.get("source_quality_score", 0)
       - 1.2 * noise
   )




def quote_similarity(a: str, b: str) -> float:
   aw = set(tokenize_for_matching(a))
   bw = set(tokenize_for_matching(b))


   if not aw or not bw:
       return 0.0


   return len(aw & bw) / len(aw | bw)





# ============================================================
# 7. Quote mining + one-best-quote selection
# LLM can choose only candidate IDs, never rewrite quotes
# ============================================================


def make_quote_window(sentences, start_idx: int, window_size: int):
   """
   Build a contiguous quote window.
   """
   end_idx = min(len(sentences), start_idx + window_size)
   items = sentences[start_idx:end_idx]


   text = normalize_quote_text(" ".join(x["sentence"] for x in items))
   start_char = items[0]["start_char"] if items else -1
   end_char = items[-1]["end_char"] if items else -1


   return {
       "sentence": text,
       "start_char": start_char,
       "end_char": end_char,
       "window_size": window_size,
       "idx": start_idx,
   }




def find_high_value_anchor_indices(sentences, source: dict, max_anchors: int = 35):
   """
   Find anchor sentences likely to contain useful support.
   """
   anchors = []


   for i, item in enumerate(sentences):
       s = item["sentence"]


       if not quote_quality_filter(s):
           continue


       score = quote_claim_score(s, source)


       if score <= 0:
           continue


       anchors.append({
           **item,
           "idx": i,
           "score": score,
           "window_size": 1,
       })


   anchors.sort(key=lambda x: x["score"], reverse=True)
   return anchors[:max_anchors]




def expand_anchor_to_candidate_windows(sentences, anchor_idx: int, source: dict):
   """
   Return multiple windows around an anchor.
   The LLM judge will choose the best one.
   """
   candidates = []


   spans = [
       (anchor_idx, 1),
       (max(0, anchor_idx - 1), 2),
       (anchor_idx, 2),
       (max(0, anchor_idx - 1), 3),
       (anchor_idx, 3),
       (max(0, anchor_idx - 2), 3),
   ]


   for start_idx, size in spans:
       if start_idx >= len(sentences):
           continue


       w = make_quote_window(sentences, start_idx, size)
       text = w["sentence"]


       if not quote_quality_filter(text):
           continue


       score = quote_claim_score(text, source)


       if size == 2:
           score += 0.20
       elif size == 3:
           score += 0.10


       w["score"] = score
       w["pre_support_score"] = score
       candidates.append(w)


   return candidates




def get_candidate_quotes_smart(
   source_text: str,
   source: dict,
   top_k: int = 30,
   max_sentences: int = 1400
):
   """
   Produce a compact candidate set for one source.
   """
   sentences = split_source_into_sentences(source_text)


   if not sentences:
       return []


   if len(sentences) > 6:
       working = sentences[1:max_sentences]
       offset = 1
   else:
       working = sentences[:max_sentences]
       offset = 0


   indexed = []
   for j, item in enumerate(working):
       indexed.append({
           **item,
           "original_idx": j + offset,
       })


   anchors_local = find_high_value_anchor_indices(
       indexed,
       source,
       max_anchors=max(35, top_k * 2)
   )


   candidates = []
   seen = set()


   for anchor in anchors_local:
       anchor_idx = anchor["original_idx"]
       windows = expand_anchor_to_candidate_windows(sentences, anchor_idx, source)


       for w in windows:
           text = w["sentence"]
           key = text.lower()


           if key in seen:
               continue


           seen.add(key)
           candidates.append(w)


   candidates.sort(key=lambda x: x["score"], reverse=True)


   selected = []
   for c in candidates:
       if any(quote_similarity(c["sentence"], prev["sentence"]) > 0.72 for prev in selected):
           continue


       selected.append(c)


       if len(selected) >= top_k:
           break


   return selected




def llm_choose_single_best_quote_id(
   claim: dict,
   source: dict,
   candidates
):
   """
   LLM chooses exactly one candidate ID.
   The LLM never writes the quote text.
   """
   if not candidates:
       return None


   candidate_payload = []
   lookup = {}


   for i, c in enumerate(candidates, 1):
       qid = f"Q{i}"
       lookup[qid] = c


       candidate_payload.append({
           "id": qid,
           "quote": c["sentence"],
           "score": round(c.get("score", 0), 4),
           "window_size": c.get("window_size", 1),
       })


   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


   system = """
You are a quote selection judge for a research citation assistant.


Select the single best candidate quote ID for supporting the claim.


Rules:
- Return only one ID from the provided candidates.
- Do not rewrite, shorten, paraphrase, or invent quote text.
- Prefer the quote that most directly supports the claim.
- Prefer explanatory or formal support over generic keyword mentions.
- Avoid metadata, title text, keyword lists, author information, references, or vague background.
- If a longer quote is necessary because it contains the full explanation, choose it.
- If multiple quotes are strong, choose the one that would be best as a citation quote in an academic paper.


Return only valid JSON:
{
 "selected_id": "Q1",
 "reason": "brief explanation"
}
"""


   human = f"""
Claim:
{claim.get("claim", "")}


Support needed:
{claim.get("what_to_support", claim.get("claim", ""))}


Source title:
{source.get("title", "")}


Candidate quotes:
{json.dumps(candidate_payload, indent=2)}
"""


   try:
       response = llm.invoke([
           SystemMessage(content=system),
           HumanMessage(content=human),
       ])


       text = response.content.strip()


       if "```json" in text:
           text = text.split("```json")[1].split("```")[0].strip()
       elif "```" in text:
           text = text.split("```")[1].split("```")[0].strip()


       parsed = json.loads(text)
       selected_id = parsed.get("selected_id", "")


       if selected_id in lookup:
           chosen = lookup[selected_id]
           chosen["llm_selected"] = True
           chosen["llm_reason"] = parsed.get("reason", "")
           chosen["rerank_score"] = chosen.get("score", 0)
           return chosen


       return None


   except Exception:
       return None




def choose_single_best_quote_without_misquoting(
   claim: dict,
   source: dict,
   candidates,
   use_llm_quote_judge: bool = True
):
   """
   Final one-quote chooser.
   If LLM fails, fallback to top lexical candidate.
   """
   if not candidates:
       return None


   if use_llm_quote_judge:
       chosen = llm_choose_single_best_quote_id(
           claim=claim,
           source=source,
           candidates=candidates
       )


       if chosen:
           return chosen


   fallback = candidates[0]
   fallback["llm_selected"] = False
   fallback["llm_reason"] = ""
   fallback["rerank_score"] = fallback.get("score", 0)
   return fallback




def extract_best_quote_from_best_source_for_claim(
   claim: dict,
   sources,
   diagnostics,
   max_sources_to_try=3,
   candidates_per_source=30,
   use_llm_quote_judge=True
):
   """
   Try ranked sources in order.
   Return exactly one best quote from the first source that produces a strong quote.


   This enforces:
   - one best quote per claim
   - from the best usable source
   - no quote hallucination because final text comes from candidate object
   """
   tried_sources = 0


   for source in sources:
       if tried_sources >= max_sources_to_try:
           break


       url = source["url"]
       source_name = urlparse(url).netloc
       source_title = source["title"]
       required_terms = source.get("required_terms", [])


       source_text = fetch_webpage_text(url, max_chars=90000)


       if source_text.startswith("Error") or len(source_text) < 1200:
           diagnostics.append(f"Skipped fetch failed/short text: {source_title} | {url}")
           continue


       if source.get("need") == "application":
           req_score = required_term_score(required_terms, source_text)
           claim_terms = claim_content_terms(source.get("claim", ""), source.get("method_or_model", ""))
           claim_score = term_hit_fraction(source_text, claim_terms)


           if max(req_score, claim_score) == 0:
               diagnostics.append(f"Skipped full text missing claim/application terms: {source_title}")
               continue


       tried_sources += 1
       full_quality = source_quality_score(url, source_text)


       candidates = get_candidate_quotes_smart(
           source_text=source_text,
           source=source,
           top_k=candidates_per_source,
           max_sentences=1400
       )


       if not candidates:
           diagnostics.append(f"No quote candidates: {source_title} | {url}")
           continue


       chosen = choose_single_best_quote_without_misquoting(
           claim=claim,
           source=source,
           candidates=candidates,
           use_llm_quote_judge=use_llm_quote_judge
       )


       if not chosen:
           diagnostics.append(f"No selected quote: {source_title} | {url}")
           continue


       chosen["sentence"] = normalize_quote_text(chosen["sentence"])


       if not quote_quality_filter(chosen["sentence"]):
           diagnostics.append(f"Selected quote failed quality filter: {source_title} | {url}")
           continue


       normalized_source_text = normalize_quote_text(source_text)
       verified = chosen["sentence"] in normalized_source_text


       if not verified:
           diagnostics.append(f"Selected quote failed verification: {source_title} | {url}")
           continue


       chosen["verified"] = True
       chosen["claim_id"] = source.get("claim_id", "")
       chosen["claim"] = source.get("claim", "")
       chosen["source_name"] = source_name
       chosen["source_title"] = source_title
       chosen["url"] = url
       chosen["need"] = source.get("need", "")
       chosen["what_to_support"] = source.get("what_to_support", "")
       chosen["source_rank_score"] = source.get("rank_score", 0)
       chosen["source_quality_score"] = full_quality
       chosen["required_terms"] = required_terms
       chosen["global_quote_score"] = quote_global_score(chosen, source)


       diagnostics.append(
           f"Selected one best quote from best usable source: {source_title} | candidates={len(candidates)}"
       )


       return chosen


   return None



# ============================================================
# 8. Formatting
# One best quote per claim
# ============================================================


def format_quote_results(plan, final_quotes, diagnostics=None):
   cleaned = plan.get("cleaned_paragraph", "")


   if not final_quotes:
       msg = (
           "I found candidate sources, but could not extract verified supporting quotes.\n\n"
           f"Cleaned paragraph:\n{cleaned}"
       )
       if diagnostics:
           msg += "\n\nDiagnostics:\n" + "\n".join(diagnostics[:25])
       return msg


   quote_lookup = {q["claim_id"]: q for q in final_quotes}
   sections = []


   for claim in plan.get("claims", []):
       claim_id = claim["claim_id"]
       claim_text = claim.get("claim", "")


       lines = [
           f"Claim {claim_id}: {claim_text}",
           ""
       ]


       q = quote_lookup.get(claim_id)


       if not q:
           lines.append("No verified quote found.")
           sections.append("\n".join(lines))
           continue


       lines.append(
           f'Best quote: "{q["sentence"]}"\n'
           f'   Source title: {q["source_title"]}\n'
           f'   Source: {q["source_name"]}\n'
           f'   URL: {q["url"]}\n'
           f'   Location: chars {q["start_char"]}–{q["end_char"]}\n'
           f'   Verified: {q["verified"]}\n'
           f'   LLM selected: {q.get("llm_selected", False)}\n'
           f'   LLM reason: {q.get("llm_reason", "")}\n'
           f'   Source rank score: {q.get("source_rank_score", 0):.1f}\n'
           f'   Quote score: {q.get("score", 0):.3f}\n'
           f'   Claim term score: {q.get("claim_term_score", 0):.3f}\n'
           f'   Phrase coverage score: {q.get("phrase_coverage_score", 0):.3f}\n'
           f'   Required-term quote score: {q.get("required_term_score_quote", 0):.3f}\n'
           f'   Keyword score: {q.get("quote_keyword_score", 0):.3f}\n'
           f'   Method bonus: {q.get("method_bonus", 0):.3f}\n'
           f'   Structure bonus: {q.get("structure_bonus", 0):.3f}\n'
           f'   Density score: {q.get("density_score", 0):.3f}\n'
           f'   Noise penalty: {q.get("quote_noise_penalty", 0):.3f}\n'
           f'   Global score: {q.get("global_quote_score", 0):.3f}'
       )


       sections.append("\n".join(lines))


   return (
       f"Cleaned paragraph:\n{cleaned}\n\n"
       f"Best supporting quote for each citation claim:\n\n"
       + "\n\n" + "=" * 80 + "\n\n".join(sections)
   )






# ============================================================
# 9. Main tool
# One best quote from best usable source per claim
# ============================================================


@tool
def find_paper_quotes(query: str) -> str:
   """
   Find high-quality sources and extract one verified best quote per atomic citation claim.
   The LLM can choose only candidate IDs, so quote text cannot be hallucinated.
   """
   plan, ranked_sources = collect_web_candidates(query, num_per_query=8, verbose=False)


   if not ranked_sources:
       return (
           "No web candidates found after relevance filtering.\n\n"
           f"Cleaned paragraph:\n{plan.get('cleaned_paragraph', clean_reference_request(query))}"
       )


   diagnostics = []
   final_quotes = []


   max_sources_to_try = 3
   candidates_per_source = 30


   by_claim = {}
   for src in ranked_sources:
       by_claim.setdefault(src["claim_id"], []).append(src)


   for claim in plan.get("claims", []):
       claim_id = claim["claim_id"]
       sources = by_claim.get(claim_id, [])


       best_quote = extract_best_quote_from_best_source_for_claim(
           claim=claim,
           sources=sources,
           diagnostics=diagnostics,
           max_sources_to_try=max_sources_to_try,
           candidates_per_source=candidates_per_source,
           use_llm_quote_judge=True
       )


       if not best_quote:
           fallback_sources = collect_additional_candidates_for_claim(claim, num_per_query=8)


           if fallback_sources:
               diagnostics.append(
                   f"Fallback search used for {claim_id}: {len(fallback_sources)} additional candidates."
               )


               best_quote = extract_best_quote_from_best_source_for_claim(
                   claim=claim,
                   sources=fallback_sources,
                   diagnostics=diagnostics,
                   max_sources_to_try=max_sources_to_try,
                   candidates_per_source=candidates_per_source,
                   use_llm_quote_judge=True
               )


       if best_quote:
           final_quotes.append(best_quote)


   return format_quote_results(plan, final_quotes, diagnostics)

