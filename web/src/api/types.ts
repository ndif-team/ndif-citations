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
  authors: string[]
  venue: string | null
  year: number | null
  category: Category
  bucket: Bucket
  confidence_band: ConfidenceBand
  reason: string | null
  source: string | null
  has_image: boolean
  manual_override: boolean
  url: string | null
}

export interface PaperDetail extends PaperRow {
  abstract: string | null
  affiliations: string[] | null
  bibtex: string | null
  image: string | null
  project_url: string | null
  pdf_url: string | null
  category_confidence_band: ConfidenceBand
  reason_detail: string | null
  venue_source: string | null
  [key: string]: unknown
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
  url: string
  name: string
  repo_type: string
  stars: number | null
  description: string | null
}
