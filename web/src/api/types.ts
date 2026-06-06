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

// Run state
export type RunState = 'running' | 'awaiting_review' | 'done' | 'error' | 'cancelled'
export type RunMode = 'fresh' | 'incremental'
export type PipelineStage = 'discover' | 'enrich' | 'route' | 'process' | 'finalize'

export interface RunSummary {
  run_id: string
  state: string
  started_at: string | null
  completed_at: string | null
  total?: number
  processed?: number
}

// Full run record from GET /runs/{id} or GET /runs/active
export interface RunRecord {
  run_id: string
  state: RunState
  mode: RunMode
  started_at: string | null
  finished_at: string | null
  error: string | null
  counts: Record<string, number> | null
  events: ProgressEvent[]
  paper_candidates: PaperCandidate[]
  repo_candidates: RepoCandidate[]
}

// SSE ProgressEvent shapes
export type ProgressEventType =
  | 'stage_start'
  | 'stage_done'
  | 'source_count'
  | 'dedup'
  | 'route_summary'
  | 'item_start'
  | 'item_step'
  | 'rate_limit_wait'
  | 'awaiting_review'
  | 'merge_result'
  | 'report'
  | 'error'
  | 'cancelled'
  | 'done'

export interface ProgressEvent {
  type: ProgressEventType
  stage?: PipelineStage
  data: Record<string, unknown>
  ts: string
}

// Review gate candidates
export interface PaperCandidate {
  id: string
  title: string
  authors?: string
  venue?: string
  year?: number
  category?: string
  bucket?: string
  url?: string
  source?: string
  [key: string]: unknown
}

export interface RepoCandidate {
  id: string
  owner: string
  repo: string
  url?: string
  description?: string
  [key: string]: unknown
}

// Gate submission
export interface GatePayload {
  process_ids: string[]
  discard_ids: string[]
  edits: Record<string, Record<string, string>>
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
