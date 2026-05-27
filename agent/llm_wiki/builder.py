"""Distil LumenX raw product + company JSON into markdown wiki pages.

Output layout (under data/wiki/):
  company.md
  products/<id>.md  (one per product)

Each markdown page is partitioned by `## SectionName` headers. The builder
also returns a list of WikiChunk records (one per section) which the
retriever indexes verbatim.

Hallucination policy: every section is built from verbatim source fields.
We do not paraphrase pricing, refund, or cancellation text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
WIKI_DIR = REPO_ROOT / "data" / "wiki"


@dataclass
class WikiChunk:
    chunk_id: str
    product_id: str | None  # None for company-wide pages
    section: str
    title: str
    text: str
    source_path: str  # relative to repo root


# ---------- field rendering helpers ----------

def _render_pricing_tiers(pricing: dict[str, Any]) -> str:
    """Per-tier price lines only. Pricing schema varies per product — render
    whatever fields are present, never invent."""
    lines: list[str] = []
    for tier, details in pricing.items():
        if isinstance(details, dict):
            monthly = details.get("monthly_usd")
            extras = {k: v for k, v in details.items() if k != "monthly_usd"}
            parts: list[str] = []
            if monthly is not None:
                parts.append(f"${monthly}/month")
            for k, v in extras.items():
                parts.append(f"{k.replace('_', ' ')}: {v}")
            line = f"- **{tier}:** " + ", ".join(parts) if parts else f"- **{tier}:** (see provider)"
        else:
            line = f"- **{tier}:** {details}"
        lines.append(line)
    return "\n".join(lines)


def _render_list(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items)


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ---------- product page ----------

def build_product_page(product: dict[str, Any]) -> tuple[str, list[WikiChunk]]:
    pid = product["id"]
    name = product["name"]
    source_rel = f"data/wiki/products/{pid}.md"

    # Header block (not a section itself; part of every chunk's context)
    header = f"# {name} (id: {pid})\n> Category: {product.get('category', '—')}\n\n"

    # Sections: (title, slug, md_body, chunk_body_override or None)
    # Most sections use the same text for markdown and chunk. Pricing
    # differs: the markdown shows the annual-discount line but the chunk
    # omits it — that fact lives once, in the company-wide annual_discount
    # chunk, so repeating it across 20 product chunks doesn't pollute
    # retrieval for company-wide discount questions.
    sections: list[tuple[str, str, str, str | None]] = []

    identity_body = (
        f"**Tagline:** {product.get('tagline', '')}\n\n"
        f"{product.get('description', '')}".strip()
    )
    sections.append(("Identity", "identity", identity_body, None))

    sections.append((
        "Target audience",
        "target_audience",
        product.get("target_audience", "").strip(),
        None,
    ))

    pricing_tiers = _render_pricing_tiers(product.get("pricing", {}))
    annual_pct = product.get("annual_discount_pct")
    pricing_md = (
        pricing_tiers
        + (f"\n\nAnnual discount: {annual_pct}% off when billed annually." if annual_pct else "")
    )
    sections.append(("Pricing", "pricing", pricing_md, pricing_tiers))

    sections.append((
        "Refund policy",
        "refund",
        product.get("refund", "").strip(),
        None,
    ))

    sections.append((
        "Cancellation policy",
        "cancellation",
        product.get("cancellation", "").strip(),
        None,
    ))

    sections.append((
        "Features",
        "features",
        _render_list(product.get("features", [])),
        None,
    ))

    sections.append((
        "Integrations",
        "integrations",
        ", ".join(product.get("integrations", [])) or "—",
        None,
    ))

    sla = product.get("support_sla_hours")
    sections.append((
        "Support SLA",
        "support_sla",
        f"First response within {sla} hours." if sla else "—",
        None,
    ))

    md_parts = [header]
    chunks: list[WikiChunk] = []
    for title, slug, md_body, chunk_body in sections:
        md_parts.append(f"## {title}\n\n{md_body}\n")
        body_for_chunk = chunk_body if chunk_body is not None else md_body
        chunk_text = f"{name} — {title}\n\n{body_for_chunk}".strip()
        chunks.append(
            WikiChunk(
                chunk_id=f"{pid}__{slug}",
                product_id=pid,
                section=slug,
                title=f"{name} — {title}",
                text=chunk_text,
                source_path=source_rel,
            )
        )

    return "\n".join(md_parts), chunks


# ---------- company page ----------

# Index-time query expansion: each company section gets a "Topics" line so
# the embedder can match casual paraphrases ("non-profit discount" vs the
# verbatim source which calls it "Lumenx Campus" or "education program").
# Verbatim policy text is NEVER changed — these are additive synonyms only.
_COMPANY_TOPICS: dict[str, str] = {
    "about": "company overview, lumenx, contact, support hours, location, founded",
    "refund_window": "refund period, money back, return policy, refund days, days to refund",
    "free_trial": "trial, free trial, try before buy, evaluation period",
    "annual_discount": "yearly billing, annual subscription, save with annual, yearly discount",
    "startup_program": (
        "startup discount, founder discount, seed-stage, early-stage company, "
        "lumenx liftoff, YC, accelerator"
    ),
    "education_program": (
        "non-profit discount, NGO discount, charity discount, student discount, "
        "teacher discount, school, university, education, academic, lumenx campus"
    ),
    "bundle": (
        "multiple products, suite, multi-product bundle, bulk discount, "
        "combined plan, lumenx suite"
    ),
}


def build_company_page(company: dict[str, Any]) -> tuple[str, list[WikiChunk]]:
    source_rel = "data/wiki/company.md"
    name = company.get("name", "Lumenx")
    header = f"# {name} — company-wide policies\n> {company.get('tagline', '')}\n\n"

    sections: list[tuple[str, str, str]] = []

    sections.append((
        "About",
        "about",
        (
            f"{company.get('description', '')}\n\n"
            f"- Founded: {company.get('founded', '—')}\n"
            f"- Headquarters: {company.get('headquarters', '—')}\n"
            f"- Support email: {company.get('support_email', '—')}\n"
            f"- Support hours: {company.get('support_hours', '—')}\n"
            f"- Billing currency: {company.get('billing_currency', '—')}"
        ).strip(),
    ))

    sections.append((
        "Refund window",
        "refund_window",
        f"Standard refund window: {company.get('refund_window_days', '—')} days from first purchase.",
    ))

    sections.append((
        "Free trial",
        "free_trial",
        f"Free trial duration: {company.get('free_trial_days', '—')} days.",
    ))

    sections.append((
        "Annual discount",
        "annual_discount",
        f"Annual billing discount: {company.get('annual_discount_pct', '—')}% off the monthly rate.",
    ))

    startup = company.get("startup_program") or {}
    sections.append((
        "Startup program",
        "startup_program",
        (
            f"**{startup.get('name', 'Startup program')}**\n\n"
            f"- Eligibility: {startup.get('eligibility', '—')}\n"
            f"- Discount: {startup.get('discount_pct', '—')}%\n"
            f"- Duration: {startup.get('duration_months', '—')} months"
        ),
    ))

    edu = company.get("education_program") or {}
    sections.append((
        "Education / non-profit program",
        "education_program",
        (
            f"**{edu.get('name', 'Education program')}**\n\n"
            f"- Eligibility: {edu.get('eligibility', '—')}\n"
            f"- Discount: {edu.get('discount_pct', '—')}%"
        ),
    ))

    bundle = company.get("bundle") or {}
    sections.append((
        "Bundle discount",
        "bundle",
        (
            f"**{bundle.get('name', 'Bundle')}**\n\n"
            f"{bundle.get('description', '')}\n\n"
            f"Discount: {bundle.get('discount_pct', '—')}%"
        ).strip(),
    ))

    md_parts = [header]
    chunks: list[WikiChunk] = []
    for title, slug, body in sections:
        md_parts.append(f"## {title}\n\n{body}\n")
        topics = _COMPANY_TOPICS.get(slug, "")
        topic_line = f"\n\nTopics: {topics}" if topics else ""
        chunk_text = f"{name} (company-wide) — {title}\n\n{body}{topic_line}".strip()
        chunks.append(
            WikiChunk(
                chunk_id=f"_company__{slug}",
                product_id=None,
                section=slug,
                title=f"{name} (company-wide) — {title}",
                text=chunk_text,
                source_path=source_rel,
            )
        )
    return "\n".join(md_parts), chunks


# ---------- top-level build ----------

def build_wiki() -> list[WikiChunk]:
    """Load cached raw data, write markdown to data/wiki/, return all chunks."""
    products_path = RAW_DIR / "products.json"
    if not products_path.exists():
        raise FileNotFoundError(
            f"{products_path} not found. Run `python -m scripts.pull_export` first."
        )

    raw = json.loads(products_path.read_text(encoding="utf-8"))
    products: list[dict[str, Any]] = raw["products"]
    company: dict[str, Any] = raw["company"]

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    (WIKI_DIR / "products").mkdir(parents=True, exist_ok=True)

    all_chunks: list[WikiChunk] = []

    company_md, company_chunks = build_company_page(company)
    (WIKI_DIR / "company.md").write_text(company_md, encoding="utf-8")
    all_chunks.extend(company_chunks)

    for product in products:
        md, chunks = build_product_page(product)
        (WIKI_DIR / "products" / f"{product['id']}.md").write_text(md, encoding="utf-8")
        all_chunks.extend(chunks)

    return all_chunks
