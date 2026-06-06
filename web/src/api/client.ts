import type { StatsResponse, PaperRow, PaperDetail, RunSummary, RepoRow, Bucket, SortOption, RunRecord, RunMode, GatePayload } from './types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw Object.assign(new Error(`API error ${res.status}: ${text}`), { status: res.status })
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

export async function fetchRun(runId: string): Promise<RunRecord> {
  return get<RunRecord>(`/runs/${encodeURIComponent(runId)}`)
}

export interface ActiveRunResponse {
  active: RunRecord | null
}

export async function fetchActiveRun(): Promise<ActiveRunResponse> {
  try {
    return await get<ActiveRunResponse>('/runs/active')
  } catch {
    return { active: null }
  }
}

export interface StartRunPayload {
  mode: RunMode
  skip_papers?: boolean
  skip_github?: boolean
}

export async function startRun(payload: StartRunPayload): Promise<{ run_id: string; state: string }> {
  return post<{ run_id: string; state: string }>('/runs', payload)
}

export async function cancelRun(runId: string): Promise<void> {
  await post(`/runs/${encodeURIComponent(runId)}/cancel`)
}

export async function submitGate(runId: string, payload: GatePayload): Promise<void> {
  await post(`/runs/${encodeURIComponent(runId)}/gate`, payload)
}

export async function fetchRepos(): Promise<RepoRow[]> {
  return get<RepoRow[]>('/repos')
}
