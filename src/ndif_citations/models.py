"""Pydantic models for the NDIF citation tracking pipeline."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    """Single 4-way classification (replaces DetailCategory)."""
    USES_NDIF = "uses_ndif"
    USES_NNSIGHT = "uses_nnsight"
    REFERENCING = "referencing"
    UNCLASSIFIED = "unclassified"


class Bucket(str, Enum):
    """3-bucket placement for the output pipeline."""
    PENDING = "pending"
    VERIFIED = "verified"
    DISCARDED = "discarded"


class PaperReason(str, Enum):
    """Reason a paper was placed in pending or discarded."""
    # Pending reasons
    OPENALEX_SOURCE = "openalex_source"
    LOW_CONFIDENCE = "low_confidence"
    UNCLASSIFIED_NO_KEYWORDS = "unclassified_no_keywords"
    UNCLASSIFIED_LLM = "unclassified_llm"
    STUB_METADATA = "stub_metadata"
    # Discarded reasons
    ZERO_PDF_HITS = "zero_pdf_hits"
    MANUAL_DISCARD = "manual_discard"
    # Manual curator override
    MANUAL_DEMOTE = "manual_demote"


class DiscoverySource(str, Enum):
    """How the paper was discovered."""
    S2_CITATION = "s2_citation"
    OPENALEX_FULLTEXT = "openalex_fulltext"
    GITHUB_DEPENDENT = "github_dependent"
    SCHOLAR = "scholar"
    MANUAL_ADD = "manual_add"
    # Note: arXiv API is used for enrichment, not discovery


class DiscoveredPaper(BaseModel):
    """Internal representation with all metadata fields."""

    # Core identifiers
    title: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    s2_paper_id: Optional[str] = None
    openalex_id: Optional[str] = None

    # Metadata
    authors: str = ""  # Comma-separated
    affiliations: str = ""  # Comma-separated institutions
    venue: str = ""
    venue_source: Optional[
        Literal[
            "doi_prefix",            # deterministic DOI prefix decode
            "arxiv_comment_parsed",  # _parse_arxiv_comment matched journal_ref/comment
            "openalex",              # OpenAlex display_name
            "s2",                    # S2 publicationVenue.name
            "crossref",              # CrossRef container_title
            "openreview",            # OpenReview venue
            "existing",              # carried over from paper.venue
            "title_search",          # title-search fallback (no identifiers)
            "fallback",              # "ArXiv {year}" default
        ]
    ] = None
    year: int = 0
    publication_date: Optional[date] = None
    peer_reviewed: Optional[bool] = None
    venue_type: Optional[str] = None  # "conference", "workshop", "journal", "preprint"

    # URLs
    url: str = ""  # Landing page
    pdf_url: Optional[str] = None

    # Content
    abstract: Optional[str] = None
    bibtex: Optional[str] = None

    # Pipeline outputs
    description: str = ""  # LLM-generated summary
    category: Category = Category.REFERENCING
    category_confidence: float = 0.0
    image: Optional[str] = None  # Path like "/images/Slug.png"

    # 3-bucket placement
    bucket: Bucket = Bucket.VERIFIED
    reason: Optional[PaperReason] = None
    reason_detail: Optional[str] = None  # free-text supplement to `reason`

    # Tracking
    source: DiscoverySource = DiscoverySource.S2_CITATION
    date_discovered: datetime = Field(default_factory=datetime.now)
    manual_override: bool = False  # If true, preserve description/category/bucket on merge
    github_repo_url: Optional[str] = None
    # Cross-link tier (1=BibTeX, 2=Citation section, 3=single ID, 4=most-recent; None when not cross-linked)
    linked_paper_tier: Optional[int] = None

    # Change detection
    content_hash: str = ""  # SHA256(title + "::" + abstract)[:16]

    # Processing flags
    has_summary: bool = False
    has_classification: bool = False
    has_thumbnail: bool = False
    has_affiliations: bool = False

    # Classification diagnostics
    unclassified_reason: Optional[
        Literal[
            "no_evidence_extractable",
            "no_keywords_anywhere",
            "llm_returned_unclassified",
            "llm_unparseable",
        ]
    ] = None

    # Pre-filter classification path (set when a pre-filter classifies without LLM; None otherwise)
    classification_signal: Optional[str] = None

    # Routing metadata for debugging
    processing_bucket: str = "UNKNOWN"  # NEW, REPROCESS, FILL_GAPS, SKIP, PROTECTED

    def compute_hash(self) -> str:
        """Compute content hash for change detection."""
        content = f"{self.title}::{self.abstract or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def model_post_init(self, __context):
        """Auto-compute hash on creation if not set."""
        if not self.content_hash:
            self.content_hash = self.compute_hash()
        if not hasattr(self, 'has_summary') or self.has_summary is None:
            self.has_summary = bool(self.description)
        if not hasattr(self, 'has_classification') or self.has_classification is None:
            self.has_classification = self.category != Category.REFERENCING or self.category_confidence > 0
        if not hasattr(self, 'has_thumbnail') or self.has_thumbnail is None:
            self.has_thumbnail = bool(self.image)
        if not hasattr(self, 'has_affiliations') or self.has_affiliations is None:
            self.has_affiliations = bool(self.affiliations)

    def to_website_dict(self) -> dict:
        """Export as website-compatible dict matching ResearchPaper TS interface.

        Returns category as "uses_ndif", "uses_nnsight", or "referencing" (never "unclassified"
        — unclassified papers belong in pending and must not reach this path).
        """
        result: dict = {
            "title": self.title,
            "authors": self.authors,
            "venue": self.venue,
            "year": self.year,
            "url": self.url,
            "description": self.description,
            "category": self.category.value,
        }
        if self.image:
            result["image"] = self.image
        if self.github_repo_url:
            result["github_repo_url"] = self.github_repo_url
        return result

    def to_full_dict(self) -> dict:
        """Export all fields for the full JSON output."""
        return self.model_dump(mode="json")

    def merge_key(self) -> str:
        """Return best key for deduplication: arxiv_id > doi > normalized title."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi}"
        return f"title:{self.title.lower().strip()}"


class DiscoveredRepo(BaseModel):
    """Internal representation of a GitHub repository discovered via dependents."""

    # Identity
    owner: str
    repo: str
    url: str
    description: Optional[str] = None

    # Activity
    stars: Optional[int] = None
    forks: Optional[int] = None
    last_commit: Optional[date] = None
    archived: bool = False
    is_fork: bool = False

    # Tech (provisional)
    language: Optional[str] = None
    license: Optional[str] = None
    topics: list[str] = Field(default_factory=list)

    # Linkage (minimal — no title/arxiv_id copies)
    readme_arxiv_ids: list[str] = Field(default_factory=list)
    linked_paper_url: Optional[str] = None  # bare arXiv URL, e.g. "https://arxiv.org/abs/2407.14561"
    # Linkage tier (1=BibTeX, 2=Citation section, 3=single post-2020, 4=most-recent-of-many; None when unlinked)
    linked_paper_tier: Optional[int] = None

    # Classification
    category: Category = Category.USES_NNSIGHT
    classification_reason: str = "github_dependent"

    # Classification metadata
    repo_type: str = "experiment"  # values: "research", "course", "experiment"
    parent_full_name: Optional[str] = None  # set to parent.full_name for forks, None otherwise

    # Persistence
    content_hash: str = ""
    manual_override: bool = False
    has_metadata: bool = False
    has_classification: bool = False
    processing_bucket: str = ""

    def merge_key(self) -> str:
        """Return key for deduplication: owner/repo."""
        return f"{self.owner}/{self.repo}"

    def compute_content_hash(self) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(
            ((self.description or "") + "::" + str(self.last_commit) + "::" + str(self.archived)).encode()
        ).hexdigest()[:16]

    def model_post_init(self, __context):
        """Auto-compute hash on creation if not set."""
        if not self.content_hash:
            self.content_hash = self.compute_content_hash()

    def to_website_dict(self) -> dict:
        """Export slim dict for github-repos.json."""
        return {
            "owner": self.owner,
            "repo": self.repo,
            "url": self.url,
            "description": self.description,
            "stars": self.stars,
            "forks": self.forks,
            "last_commit": self.last_commit.isoformat() if self.last_commit else None,
            "language": self.language,
            "license": self.license,
            "topics": self.topics,
            "archived": self.archived,
            "category": self.category.value,
            "linked_paper_url": self.linked_paper_url,
            "linked_paper_tier": self.linked_paper_tier,
            "repo_type": self.repo_type,
            "parent_full_name": self.parent_full_name,
        }

    def to_full_dict(self) -> dict:
        """Export all fields."""
        return self.model_dump(mode="json")


class PipelineRun(BaseModel):
    """Metadata about a pipeline execution."""
    run_date: datetime = Field(default_factory=datetime.now)
    s2_citations_found: int = 0
    openalex_found: int = 0
    scholar_found: int = 0
    github_dependents_found: int = 0
    total_unique: int = 0
    new_papers: int = 0
    updated_papers: int = 0
    existing_papers: int = 0
    thumbnails_extracted: int = 0
    thumbnails_missing: int = 0
    errors: list[str] = Field(default_factory=list)
    low_confidence: list[str] = Field(default_factory=list)
    missing_thumbnails: list[str] = Field(default_factory=list)

    # Routing bucket counts
    bucket_new: int = 0
    bucket_reprocess: int = 0
    bucket_fill_gaps: int = 0
    bucket_skip: int = 0
    bucket_protected: int = 0
    unclassified_count: int = 0  # Papers we couldn't classify (PDF unavailable)

    # Auto-recovery tracking (populated during merge)
    auto_promoted: list[str] = Field(default_factory=list)   # titles promoted pending→verified
    auto_demoted: list[str] = Field(default_factory=list)    # titles demoted verified→pending
