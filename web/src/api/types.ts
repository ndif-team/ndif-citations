// API response types matching the FastAPI backend

export type Bucket = 'verified' | 'pending' | 'discarded'
export type Category = 'uses_ndif' | 'uses_nnsight' | 'referencing' | 'unclassified'
export type ConfidenceBand = 'CERTAIN' | 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'
export type SortOption = 'year_desc' | 'year_asc' | 'title'

export interface StatsResponse {
  papers: {
    verified: number
    pending: number
    discarded: number
    total: number
  }
  repos: {
    research: number
    course: number
    experiment: number
    total: number
  }
  categories: {
    uses_ndif: number
    uses_nnsight: number
    referencing: number
    unclassified: number
  }
}

export interface PaperRow {
  id: string
  title: string
  authors: string
  venue: string
  year: number
  category: Category
  bucket: Bucket
  confidence_band: ConfidenceBand
  reason: string | null
  source: string
  has_image: boolean
  manual_override: boolean
  url: string
}

export interface PaperDetail {
  id: string
  title: string
  arxiv_id: string | null
  doi: string | null
  s2_paper_id: string | null
  openalex_id: string | null
  authors: string
  affiliations: string
  venue: string
  venue_source: string | null
  year: number
  publication_date: string | null
  peer_reviewed: boolean | null
  venue_type: string | null
  url: string
  pdf_url: string | null
  abstract: string | null
  bibtex: string | null
  description: string
  category: Category
  category_confidence: number
  category_confidence_band: ConfidenceBand
  image: string | null
  bucket: Bucket
  reason: string | null
  reason_detail: string | null
  source: string
  date_discovered: string
  manual_override: boolean
  project_url: string | null
  linked_paper_tier: number | null
  content_hash: string
  has_summary: boolean
  has_classification: boolean
  has_thumbnail: boolean
  has_affiliations: boolean
  unclassified_reason: string | null
  classification_signal: string | null
  processing_bucket: string
  confidence_band: ConfidenceBand
  has_image: boolean
}

export interface RunSummary {
  run_id: string
  state: string
  started_at: string | null
  completed_at: string | null
  total?: number
  processed?: number
}

export interface RepoRow {
  id: string
  owner: string
  repo: string
  url: string
  description: string | null
  stars: number | null
  forks: number | null
  language: string | null
  repo_type: string
  category: string
  linked_paper_url: string | null
  last_commit: string | null
  manual_override: boolean
}
