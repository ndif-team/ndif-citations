import type { StatsResponse, PaperRow, PaperDetail, RunSummary, RepoRow, Bucket, SortOption } from './types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

export async function fetchStats(): Promise<StatsResponse> {
  return get<StatsResponse>('/stats')
}

export interface PapersParams {
  bucket?: Bucket | ''
  q?: string
  sort?: SortOption
}

export async function fetchPapers(params: PapersParams = {}): Promise<PaperRow[]> {
  const sp = new URLSearchParams()
  if (params.bucket) sp.set('bucket', params.bucket)
  if (params.q) sp.set('q', params.q)
  if (params.sort) sp.set('sort', params.sort)
  const qs = sp.toString() ? `?${sp.toString()}` : ''
  return get<PaperRow[]>(`/papers${qs}`)
}

export async function fetchPaper(id: string): Promise<PaperDetail> {
  return get<PaperDetail>(`/papers/${encodeURIComponent(id)}`)
}

export async function fetchRuns(): Promise<RunSummary[]> {
  return get<RunSummary[]>('/runs')
}

export async function fetchActiveRun(): Promise<RunSummary | null> {
  try {
    return await get<RunSummary>('/runs/active')
  } catch {
    return null
  }
}

export async function fetchRepos(): Promise<RepoRow[]> {
  return get<RepoRow[]>('/repos')
}
