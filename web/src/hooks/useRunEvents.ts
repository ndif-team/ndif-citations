import { useEffect, useRef, useCallback, useReducer } from 'react'
import type { ProgressEvent, PipelineStage, PaperCandidate, RepoCandidate, RunState } from '@/api/types'

export interface CurrentItem {
  idx: number
  total: number
  title: string
  bucket: string
  /** 1-based index among real-work papers (excludes skipped no-ops) — F-012 */
  workIdx?: number
  /** count of real-work papers (the honest "Processing X/workTotal" denominator) */
  workTotal?: number
}

export interface RunEventState {
  events: ProgressEvent[]
  /** Keyed by stage — last event for each stage type */
  latestByStage: Partial<Record<PipelineStage, ProgressEvent>>
  /** Stage completion status derived from stage_start / stage_done */
  stageStatus: Partial<Record<PipelineStage, 'active' | 'done'>>
  currentItem: CurrentItem | null
  /** Count of papers skipped as already-complete during the process stage (F-012) */
  skippedCount: number
  /** Seconds remaining in rate-limit wait, or null */
  rateLimitWait: number | null
  /** Source of the active rate-limit wait (e.g. "LLM summary", "OpenAlex", "S2"), or null */
  rateLimitLabel: string | null
  candidates: { papers: PaperCandidate[]; repos: RepoCandidate[] } | null
  /** Per-bucket routing breakdown captured at the gate, for the work-preview (F-012) */
  routeBreakdown: Record<string, number> | null
  /** Derived state, updated as events arrive */
  derivedState: RunState | null
  error: string | null
  ended: boolean
}

type Action =
  | { type: 'event'; event: ProgressEvent }
  | { type: 'rate_tick' }
  | { type: 'ended' }
  | { type: 'reset' }

function initialState(): RunEventState {
  return {
    events: [],
    latestByStage: {},
    stageStatus: {},
    currentItem: null,
    skippedCount: 0,
    rateLimitWait: null,
    rateLimitLabel: null,
    candidates: null,
    routeBreakdown: null,
    derivedState: null,
    error: null,
    ended: false,
  }
}

function reducer(state: RunEventState, action: Action): RunEventState {
  switch (action.type) {
    case 'event': {
      const e = action.event
      const events = [...state.events, e]
      const latestByStage = { ...state.latestByStage }
      const stageStatus = { ...state.stageStatus }
      let currentItem = state.currentItem
      let skippedCount = state.skippedCount
      let rateLimitWait = state.rateLimitWait
      let rateLimitLabel = state.rateLimitLabel
      let candidates = state.candidates
      let routeBreakdown = state.routeBreakdown
      let derivedState = state.derivedState
      let error = state.error

      // Update stage tracking
      if ((e.type === 'stage_start' || e.type === 'stage_done') && e.stage) {
        latestByStage[e.stage] = e
        stageStatus[e.stage] = e.type === 'stage_done' ? 'done' : 'active'
      }

      if (e.type === 'item_start') {
        const d = e.data as { idx?: number; total?: number; title?: string; bucket?: string; work_idx?: number; work_total?: number }
        currentItem = {
          idx: typeof d.idx === 'number' ? d.idx : 0,
          total: typeof d.total === 'number' ? d.total : 0,
          title: typeof d.title === 'string' ? d.title : '',
          bucket: typeof d.bucket === 'string' ? d.bucket : '',
          workIdx: typeof d.work_idx === 'number' ? d.work_idx : undefined,
          workTotal: typeof d.work_total === 'number' ? d.work_total : undefined,
        }
      }

      // Already-complete papers are copied through without LLM work — count them
      // separately so the headline reflects real work, not the routed total (F-012).
      if (e.type === 'item_skip') {
        skippedCount = skippedCount + 1
      }

      if (e.type === 'rate_limit_wait') {
        const d = e.data as { seconds?: number; label?: string }
        rateLimitWait = typeof d.seconds === 'number' ? d.seconds : null
        rateLimitLabel = typeof d.label === 'string' && d.label ? d.label : null
      }

      if (e.type === 'awaiting_review') {
        const d = e.data as { paper_candidates?: PaperCandidate[]; repo_candidates?: RepoCandidate[]; route_breakdown?: Record<string, number> }
        candidates = {
          papers: Array.isArray(d.paper_candidates) ? d.paper_candidates : [],
          repos: Array.isArray(d.repo_candidates) ? d.repo_candidates : [],
        }
        if (d.route_breakdown && typeof d.route_breakdown === 'object') {
          routeBreakdown = d.route_breakdown
        }
        derivedState = 'awaiting_review'
      }

      if (e.type === 'done') derivedState = 'done'
      if (e.type === 'error') {
        derivedState = 'error'
        const d = e.data as { message?: string }
        error = typeof d.message === 'string' ? d.message : 'Unknown error'
      }
      if (e.type === 'cancelled') derivedState = 'cancelled'

      // Clear rate_limit_wait once processing resumes (stage_start for process after a wait)
      if (e.type === 'stage_start' && e.stage === 'process') {
        rateLimitWait = null
        rateLimitLabel = null
      }
      if (e.type === 'item_start') {
        // Also clear on next item start
        rateLimitWait = null
        rateLimitLabel = null
      }

      return { ...state, events, latestByStage, stageStatus, currentItem, skippedCount, rateLimitWait, rateLimitLabel, candidates, routeBreakdown, derivedState, error }
    }
    case 'rate_tick': {
      if (state.rateLimitWait === null || state.rateLimitWait <= 0) {
        return { ...state, rateLimitWait: null, rateLimitLabel: null }
      }
      return { ...state, rateLimitWait: state.rateLimitWait - 1 }
    }
    case 'ended':
      return { ...state, ended: true }
    case 'reset':
      return initialState()
    default:
      return state
  }
}

export function useRunEvents(runId: string | null) {
  const [state, dispatch] = useReducer(reducer, undefined, initialState)
  const esRef = useRef<EventSource | null>(null)
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Countdown ticker for rate_limit_wait
  const startTick = useCallback(() => {
    if (tickRef.current) return
    tickRef.current = setInterval(() => {
      dispatch({ type: 'rate_tick' })
    }, 1000)
  }, [])

  const stopTick = useCallback(() => {
    if (tickRef.current) {
      clearInterval(tickRef.current)
      tickRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!runId) {
      dispatch({ type: 'reset' })
      return
    }

    dispatch({ type: 'reset' })

    const es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`)
    esRef.current = es

    es.onmessage = (e: MessageEvent) => {
      try {
        const event = JSON.parse(e.data as string) as ProgressEvent
        dispatch({ type: 'event', event })
        // Start countdown if rate limit wait
        if (event.type === 'rate_limit_wait') {
          startTick()
        } else if (event.type === 'item_start' || event.type === 'stage_start') {
          stopTick()
        }
      } catch {
        // Ignore malformed events — don't crash
      }
    }

    es.addEventListener('end', () => {
      dispatch({ type: 'ended' })
      es.close()
      stopTick()
    })

    es.onerror = () => {
      // EventSource will auto-reconnect on transient errors;
      // only close if we've already ended.
    }

    return () => {
      es.close()
      esRef.current = null
      stopTick()
    }
  }, [runId, startTick, stopTick])

  return state
}
