"""Pydantic models for the NDIF citation tracking pipeline."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DetailCategory(str, Enum):
    """Four-way internal categorization."""
    USES_NDIF = "uses_ndif"
    USES_NNSIGHT = "uses_nnsight"
    REFERENCING = "referencing"
    UNCLASSIFIED = "unclassified"  # PDF unavailable or insufficient evidence


class WebsiteCategory(str, Enum):
    """Three-way category for the website JSON."""
    USED_NNSIGHT = "used-nnsight"
    USED_NDIF = "used-ndif"
    REFERENCING = "referencing"


class DiscoverySource(str, Enum):
    """How the paper was discovered."""
    S2_CITATION = "s2_citation"
    OPENALEX_FULLTEXT = "openalex_fulltext"
    GITHUB_DEPENDENT = "github_dependent"
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
    detail_category: DetailCategory = DetailCategory.REFERENCING
    category_confidence: float = 0.0
    image: Optional[str] = None  # Path like "/images/Slug.png"

    # Tracking
    source: DiscoverySource = DiscoverySource.S2_CITATION
    date_discovered: datetime = Field(default_factory=datetime.now)
    manual_override: bool = False  # If true, preserve description/category on merge
    github_repo_url: Optional[str] = None

    # Change detection 
    content_hash: str = ""  # SHA256(title + "::" + abstract)[:16]

    # Processing flags 
    has_summary: bool = False
    has_classification: bool = False
    has_thumbnail: bool = False

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
        # Set processing flags based on existing data
        if not hasattr(self, 'has_summary') or self.has_summary is None:
            self.has_summary = bool(self.description)
        if not hasattr(self, 'has_classification') or self.has_classification is None:
            self.has_classification = self.detail_category != DetailCategory.REFERENCING or self.category_confidence > 0
        if not hasattr(self, 'has_thumbnail') or self.has_thumbnail is None:
            self.has_thumbnail = bool(self.image)

    @property
    def website_category(self) -> WebsiteCategory:
        """Map three-way internal category 1-to-1 to website category."""
        if self.detail_category == DetailCategory.USES_NNSIGHT:
            return WebsiteCategory.USED_NNSIGHT
        if self.detail_category == DetailCategory.USES_NDIF:
            return WebsiteCategory.USED_NDIF
        return WebsiteCategory.REFERENCING

    def to_website_dict(self) -> dict:
        """Export as website-compatible dict matching ResearchPaper TS interface."""
        result: dict = {
            "title": self.title,
            "authors": self.authors,
            "venue": self.venue,
            "year": self.year,
            "url": self.url,
            "description": self.description,
            "category": self.website_category.value,
        }
        if self.image:
            result["image"] = self.image
        return result

    def to_full_dict(self) -> dict:
        """Export all fields for the full JSON output."""
        data = self.model_dump(mode="json")
        data["website_category"] = self.website_category.value
        return data

    def merge_key(self) -> str:
        """Return best key for deduplication: arxiv_id > doi > normalized title."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi}"
        return f"title:{self.title.lower().strip()}"


class ResearchPaper(BaseModel):
    """Website-facing model matching the TypeScript ResearchPaper interface (3 categories)."""
    title: str
    authors: str
    venue: str
    year: int
    url: str
    image: Optional[str] = None
    description: str
    category: WebsiteCategory  # used-nnsight, used-ndif, or referencing


class PipelineRun(BaseModel):
    """Metadata about a pipeline execution."""
    run_date: datetime = Field(default_factory=datetime.now)
    s2_citations_found: int = 0
    openalex_found: int = 0
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
