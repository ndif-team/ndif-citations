import type { StatsResponse, PaperRow, PaperDetail, RunSummary, RepoRow, RepoDetail, RepoType, RepoSortOption, Bucket, SortOption, RunRecord, RunMode, GatePayload, ReprocessResponse, SettingsResponse, SettingsPatch, VenuesResponse, PublishTargetResponse, PublishDryRunResponse, PublishResponse } from './types'

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

async function put<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw Object.assign(new Error(`API error ${res.status}: ${text}`), { status: res.status })
  }
  return res.json() as Promise<T>
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
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

export interface ReposParams {
  repo_type?: RepoType | ''
  q?: string
  sort?: RepoSortOption
}

export async function fetchRepos(params: ReposParams = {}): Promise<RepoRow[]> {
  const sp = new URLSearchParams()
  if (params.repo_type) sp.set('repo_type', params.repo_type)
  if (params.q) sp.set('q', params.q)
  if (params.sort) sp.set('sort', params.sort)
  const qs = sp.toString() ? `?${sp.toString()}` : ''
  return get<RepoRow[]>(`/repos${qs}`)
}

export async function fetchRepo(owner: string, repo: string): Promise<RepoDetail> {
  return get<RepoDetail>(`/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`)
}

export interface RepoEditPayload {
  repo_type?: RepoType
  linked_paper_url?: string | null
  description?: string | null
}

export async function editRepo(owner: string, repo: string, payload: RepoEditPayload): Promise<RepoDetail> {
  return patch<RepoDetail>(`/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`, payload)
}

export interface ExcludeRepoResponse {
  excluded: string
  remaining: number
  was_present: boolean
}

export async function excludeRepo(owner: string, repo: string): Promise<ExcludeRepoResponse> {
  return post<ExcludeRepoResponse>(`/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/exclude`)
}

// ---------------------------------------------------------------------------
// Paper mutations
// ---------------------------------------------------------------------------

export async function editPaper(id: string, fields: Record<string, string>): Promise<PaperDetail> {
  return patch<PaperDetail>(`/papers/${encodeURIComponent(id)}`, { fields })
}

export interface BucketPayload {
  bucket: Bucket
  reason?: string
  detail?: string
}

export async function setPaperBucket(id: string, payload: BucketPayload): Promise<PaperDetail> {
  return post<PaperDetail>(`/papers/${encodeURIComponent(id)}/bucket`, payload)
}

export async function uploadPaperImage(id: string, file: File): Promise<PaperDetail> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/papers/${encodeURIComponent(id)}/image`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw Object.assign(new Error(`API error ${res.status}: ${text}`), { status: res.status })
  }
  return res.json() as Promise<PaperDetail>
}

export async function reextractThumbnail(id: string): Promise<ReprocessResponse> {
  return post<ReprocessResponse>(`/papers/${encodeURIComponent(id)}/reextract-thumbnail`)
}

export async function batchReprocess(ids: string[], fields: string[]): Promise<ReprocessResponse> {
  return post<ReprocessResponse>('/papers/reprocess', { ids, fields })
}

export async function reprocessPaper(paperId: string, fields: string[]): Promise<ReprocessResponse> {
  return post<ReprocessResponse>(`/papers/${encodeURIComponent(paperId)}/reprocess`, { fields })
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export async function fetchSettings(): Promise<SettingsResponse> {
  return get<SettingsResponse>('/settings')
}

export async function putSettings(patch: SettingsPatch): Promise<SettingsResponse> {
  return put<SettingsResponse>('/settings', patch)
}

// ---------------------------------------------------------------------------
// Venues
// ---------------------------------------------------------------------------

export async function fetchVenues(): Promise<VenuesResponse> {
  return get<VenuesResponse>('/venues')
}

export async function putVenues(venues: VenuesResponse['venues']): Promise<VenuesResponse> {
  return put<VenuesResponse>('/venues', { venues })
}

// ---------------------------------------------------------------------------
// Publish
// ---------------------------------------------------------------------------

export async function fetchPublishTarget(): Promise<PublishTargetResponse> {
  return get<PublishTargetResponse>('/publish/target')
}

export async function putPublishTarget(path: string): Promise<{ publish_target: string }> {
  return put<{ publish_target: string }>('/publish/target', { path })
}

export async function runPublish(dry_run: boolean): Promise<PublishDryRunResponse | PublishResponse> {
  return post<PublishDryRunResponse | PublishResponse>('/publish', { dry_run })
}

// ---------------------------------------------------------------------------
// Paper PDF URL helper
// ---------------------------------------------------------------------------

export const paperPdfUrl = (id: string) => `/api/papers/${encodeURIComponent(id)}/pdf`

// ---------------------------------------------------------------------------
// Manual add (by URL or PDF)
// ---------------------------------------------------------------------------

export async function addPaper(url: string): Promise<{ run_id: string; state: string }> {
  return post<{ run_id: string; state: string }>('/papers/add', { url })
}

export async function addPaperPdf(args: { title: string; arxiv_id?: string; doi?: string; file: File }): Promise<{ run_id: string; state: string }> {
  const fd = new FormData()
  fd.append('title', args.title)
  if (args.arxiv_id) fd.append('arxiv_id', args.arxiv_id)
  if (args.doi) fd.append('doi', args.doi)
  fd.append('file', args.file)
  const res = await fetch(`${BASE}/papers/add-pdf`, { method: 'POST', body: fd })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw Object.assign(new Error(`API error ${res.status}: ${text}`), { status: res.status })
  }
  return res.json() as Promise<{ run_id: string; state: string }>
}

// ---------------------------------------------------------------------------
// Paper PDF upload + evidence backfill
// ---------------------------------------------------------------------------

export async function attachPdf(paperId: string, file: File): Promise<PaperDetail> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${BASE}/papers/${encodeURIComponent(paperId)}/pdf`, { method: 'POST', body: fd })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw Object.assign(new Error(`API error ${res.status}: ${text}`), { status: res.status })
  }
  return res.json() as Promise<PaperDetail>
}

export async function backfillEvidence(paperId: string): Promise<PaperDetail> {
  const res = await fetch(`${BASE}/papers/${encodeURIComponent(paperId)}/evidence`, { method: 'POST' })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw Object.assign(new Error(`API error ${res.status}: ${text}`), { status: res.status })
  }
  return res.json() as Promise<PaperDetail>
}
