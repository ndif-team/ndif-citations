import { useState, useEffect, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Play,
  Square,
  AlertCircle,
  ChevronRight,
  Check,
  Loader2,
  Clock,
  SkipForward,
  Pencil,
  X,
} from 'lucide-react'
import { toast } from 'sonner'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogFooter,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogAction,
  AlertDialogCancel,
} from '@/components/ui/alert-dialog'
import { useRuns, useActiveRun } from '@/api/hooks'
import { useRunEvents } from '@/hooks/useRunEvents'
import { startRun, cancelRun, submitGate, fetchRun, getPreflight } from '@/api/client'
import { runStateBadge, runStateLabel } from '@/lib/tokens'
import { cn } from '@/lib/utils'
import type { RunRecord, PipelineStage, PaperCandidate, RepoCandidate, RunMode, ProgressEvent } from '@/api/types'

// ─── Pipeline stage ordering ────────────────────────────────────────────────

const STAGES: { key: PipelineStage; label: string }[] = [
  { key: 'discover', label: 'Discover' },
  { key: 'enrich', label: 'Enrich' },
  { key: 'route', label: 'Route' },
  { key: 'process', label: 'Process' },
  { key: 'finalize', label: 'Finalize' },
]

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtDate(s: string | null | undefined): string {
  if (!s) return '—'
  try {
    return new Date(s).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return s
  }
}

function fmtDuration(start: string | null, end: string | null): string {
  if (!start) return ''
  const ms = (end ? new Date(end) : new Date()).getTime() - new Date(start).getTime()
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${m}m ${sec}s`
}

/** Render a ProgressEvent as a readable log line */
function eventToLine(e: ProgressEvent): string {
  try {
    const ts = e.ts ? new Date(e.ts).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : ''
    const prefix = ts ? `[${ts}]` : ''
    const d = e.data ?? {}

    switch (e.type) {
      case 'stage_start': return `${prefix} Stage start: ${e.stage ?? ''}`
      case 'stage_done': return `${prefix} Stage done: ${e.stage ?? ''}`
      case 'source_count': {
        const entries = Object.entries(d).map(([k, v]) => `${k}=${v}`).join(', ')
        return `${prefix} Source counts: ${entries}`
      }
      case 'dedup': {
        const { unique, total, dropped } = d as Record<string, unknown>
        return `${prefix} Dedup: ${unique ?? '?'} unique / ${total ?? '?'} total (${dropped ?? 0} dropped)`
      }
      case 'route_summary': {
        const { to_process, skipped } = d as Record<string, unknown>
        return `${prefix} Route: ${to_process ?? '?'} to process, ${skipped ?? '?'} skipped`
      }
      case 'item_start': {
        const { idx, total, title } = d as Record<string, unknown>
        return `${prefix} Processing ${idx ?? '?'}/${total ?? '?'}: ${title ?? ''}`
      }
      case 'item_step': {
        const { idx, step } = d as Record<string, unknown>
        return `${prefix}   Step [${step ?? '?'}] on item ${idx ?? '?'}`
      }
      case 'rate_limit_wait': {
        const { label, seconds } = d as Record<string, unknown>
        const source = typeof label === 'string' && label ? label : 'Rate limit'
        return `${prefix} ${source} cooldown: ${seconds ?? '?'}s`
      }
      case 'awaiting_review': {
        const papers = (d as { paper_candidates?: unknown[] }).paper_candidates
        const repos = (d as { repo_candidates?: unknown[] }).repo_candidates
        return `${prefix} Awaiting review: ${Array.isArray(papers) ? papers.length : 0} papers, ${Array.isArray(repos) ? repos.length : 0} repos`
      }
      case 'merge_result': {
        const entries = Object.entries(d).map(([k, v]) => `${k}=${v}`).join(', ')
        return `${prefix} Merge result: ${entries}`
      }
      case 'report': {
        const entries = Object.entries(d).map(([k, v]) => `${k}=${v}`).join(', ')
        return `${prefix} Report: ${entries}`
      }
      case 'error': {
        const { message } = d as { message?: string }
        return `${prefix} Error: ${message ?? 'unknown'}`
      }
      case 'cancelled': return `${prefix} Run cancelled`
      case 'done': return `${prefix} Run complete`
      default: return `${prefix} ${e.type}: ${JSON.stringify(d)}`
    }
  } catch {
    return `[event] ${e.type}`
  }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

/** Phase stepper bar */
function PhaseStepper({
  stageStatus,
  showReview,
}: {
  stageStatus: Partial<Record<PipelineStage, 'active' | 'done'>>
  showReview: boolean
}) {
  // Build ordered display steps (insert Review gate between route and process for incremental)
  type DisplayStep = { key: string; label: string; status: 'done' | 'active' | 'pending' }
  const steps: DisplayStep[] = []

  STAGES.forEach((s, i) => {
    steps.push({
      key: s.key,
      label: s.label,
      status: stageStatus[s.key] ?? 'pending',
    })
    // Insert review gate between route (idx 2) and process (idx 3)
    if (i === 2 && showReview) {
      steps.push({
        key: 'review',
        label: 'Review',
        status:
          stageStatus['process'] === 'done' || stageStatus['process'] === 'active'
            ? 'done'
            : stageStatus['route'] === 'done'
              ? 'active'
              : 'pending',
      })
    }
  })

  return (
    <div className="flex items-center gap-0 overflow-x-auto" role="list" aria-label="Pipeline phases">
      {steps.map((step, i) => (
        <div key={step.key} className="flex items-center" role="listitem">
          {i > 0 && (
            <ChevronRight
              className="h-3.5 w-3.5 text-muted-foreground flex-none mx-0.5"
              aria-hidden="true"
            />
          )}
          <div
            className={cn(
              'flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors motion-reduce:transition-none',
              step.status === 'done' && 'bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300',
              step.status === 'active' && 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 ring-1 ring-blue-300 dark:ring-blue-700',
              step.status === 'pending' && 'bg-muted text-muted-foreground'
            )}
          >
            {step.status === 'done' && <Check className="h-3 w-3 flex-none" aria-hidden="true" />}
            {step.status === 'active' && (
              <Loader2 className="h-3 w-3 flex-none animate-spin motion-reduce:animate-none" aria-hidden="true" />
            )}
            {step.label}
          </div>
        </div>
      ))}
    </div>
  )
}

/** Scrolling event log */
function EventLog({ events }: { events: ProgressEvent[] }) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  return (
    <div
      className="flex-1 min-h-0 overflow-y-auto rounded-md border bg-card p-3 font-mono text-xs text-muted-foreground space-y-0.5"
      aria-live="polite"
      aria-label="Run event log"
    >
      {events.length === 0 && (
        <span className="text-muted-foreground/50">Waiting for events…</span>
      )}
      {events.map((e, i) => (
        <div
          key={i}
          className={cn(
            'leading-relaxed whitespace-pre-wrap break-words',
            e.type === 'error' && 'text-destructive',
            e.type === 'done' && 'text-green-600 dark:text-green-400',
            e.type === 'cancelled' && 'text-muted-foreground',
            e.type === 'rate_limit_wait' && 'text-amber-600 dark:text-amber-400',
          )}
        >
          {eventToLine(e)}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

/** Rate-limit cooldown badge — `label` names the throttle source (LLM, OpenAlex, S2, …) */
function CooldownBadge({ seconds, label }: { seconds: number; label?: string | null }) {
  const source = label && label.trim() ? label : 'Rate limit'
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium',
        'bg-amber-50 text-amber-800 ring-1 ring-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-800',
        'animate-pulse motion-reduce:animate-none',
      )}
      role="status"
      aria-live="polite"
    >
      <Clock className="h-3.5 w-3.5 flex-none" aria-hidden="true" />
      {source} cooldown: {seconds}s
    </span>
  )
}

// ─── Edit dialog (for paper candidate field edits) ───────────────────────────

interface EditDialogProps {
  paper: PaperCandidate
  open: boolean
  onClose: () => void
  onSave: (id: string, edits: Record<string, string>) => void
  existing: Record<string, string>
}

function EditDialog({ paper, open, onClose, onSave, existing }: EditDialogProps) {
  const [title, setTitle] = useState(existing.title ?? paper.title ?? '')
  const [venue, setVenue] = useState(existing.venue ?? paper.venue ?? '')

  useEffect(() => {
    if (open) {
      setTitle(existing.title ?? (typeof paper.title === 'string' ? paper.title : ''))
      setVenue(existing.venue ?? (typeof paper.venue === 'string' ? paper.venue : ''))
    }
  }, [open, existing, paper.title, paper.venue])

  function handleSave() {
    const edits: Record<string, string> = {}
    if (title !== (paper.title ?? '')) edits.title = title
    if (venue !== (paper.venue ?? '')) edits.venue = venue
    onSave(paper.id, edits)
    onClose()
  }

  return (
    <AlertDialog open={open} onOpenChange={(v) => { if (!v) onClose() }}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Edit fields</AlertDialogTitle>
          <AlertDialogDescription>
            Override title or venue before processing this paper.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="space-y-3 mt-2">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground" htmlFor="edit-title">Title</label>
            <Input
              id="edit-title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="text-xs"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground" htmlFor="edit-venue">Venue</label>
            <Input
              id="edit-venue"
              value={venue}
              onChange={(e) => setVenue(e.target.value)}
              className="text-xs"
            />
          </div>
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={onClose}>Cancel</AlertDialogCancel>
          <button
            onClick={handleSave}
            className="inline-flex items-center justify-center rounded-md px-3 py-1.5 text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            Save
          </button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

// ─── Review Gate ─────────────────────────────────────────────────────────────

type CandidateDecision = 'process' | 'discard' | 'skip'

interface ReviewGateProps {
  runId: string
  papers: PaperCandidate[]
  repos: RepoCandidate[]
  onSubmitted: () => void
}

function ReviewGate({ runId, papers, repos, onSubmitted }: ReviewGateProps) {
  const [decisions, setDecisions] = useState<Record<string, CandidateDecision>>(() =>
    Object.fromEntries(papers.map((p) => [p.id, 'skip' as CandidateDecision]))
  )
  const [edits, setEdits] = useState<Record<string, Record<string, string>>>({})
  const [editingId, setEditingId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  function setDecision(id: string, d: CandidateDecision) {
    setDecisions((prev) => ({ ...prev, [id]: d }))
  }

  function handleEditSave(id: string, paperEdits: Record<string, string>) {
    if (Object.keys(paperEdits).length > 0) {
      setEdits((prev) => ({ ...prev, [id]: { ...(prev[id] ?? {}), ...paperEdits } }))
    } else {
      setEdits((prev) => {
        const next = { ...prev }
        delete next[id]
        return next
      })
    }
  }

  const processIds = papers.filter((p) => decisions[p.id] === 'process').map((p) => p.id)
  const discardIds = papers.filter((p) => decisions[p.id] === 'discard').map((p) => p.id)
  const skipCount = papers.length - processIds.length - discardIds.length

  async function handleSubmit() {
    setSubmitting(true)
    try {
      await submitGate(runId, { process_ids: processIds, discard_ids: discardIds, edits })
      onSubmitted()
      toast.success(`Gate submitted — processing ${processIds.length} papers`)
    } catch (err) {
      const status = (err as { status?: number }).status
      if (status === 409) {
        toast.error('Run is not awaiting review right now')
      } else if (status === 422) {
        toast.error('Invalid edits — check field values')
      } else {
        toast.error(`Failed to submit gate: ${(err as Error).message}`)
      }
    } finally {
      setSubmitting(false)
    }
  }

  const editingPaper = editingId ? papers.find((p) => p.id === editingId) ?? null : null

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">Review gate</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Decide which papers to process with LLM, discard, or skip (leave for next run).
          </p>
        </div>
      </div>

      {/* Summary bar */}
      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        <span>
          <span className="font-medium text-foreground">{processIds.length}</span> process
        </span>
        <span>·</span>
        <span>
          <span className="font-medium text-foreground">{discardIds.length}</span> discard
        </span>
        <span>·</span>
        <span>
          <span className="font-medium text-foreground">{skipCount}</span> skip
        </span>
      </div>

      {/* Papers table */}
      {papers.length === 0 ? (
        <p className="text-xs text-muted-foreground">No paper candidates to review.</p>
      ) : (
        <div className="rounded-md border overflow-hidden">
          <table className="w-full text-xs text-left" aria-label="Paper candidates review">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-3 py-2 font-medium text-muted-foreground">Title</th>
                <th className="px-3 py-2 font-medium text-muted-foreground w-24">Venue</th>
                <th className="px-3 py-2 font-medium text-muted-foreground w-28">Source</th>
                <th className="px-3 py-2 font-medium text-muted-foreground w-56 text-center">Decision</th>
                <th className="px-3 py-2 w-8" />
              </tr>
            </thead>
            <tbody>
              {papers.map((paper) => {
                const decision = decisions[paper.id] ?? 'skip'
                const paperEdits = edits[paper.id] ?? {}
                const displayTitle = paperEdits.title ?? (typeof paper.title === 'string' ? paper.title : '—')
                const displayVenue = paperEdits.venue ?? (typeof paper.venue === 'string' ? paper.venue : '—')

                return (
                  <tr key={paper.id} className="border-t border-border/50 hover:bg-muted/30 transition-colors">
                    <td className="px-3 py-2 max-w-0">
                      <span className="block truncate font-medium" title={displayTitle}>
                        {displayTitle}
                      </span>
                      {typeof paper.authors === 'string' && paper.authors && (
                        <span className="block truncate text-muted-foreground mt-0.5">{paper.authors}</span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground truncate max-w-0">
                      {displayVenue || '—'}
                    </td>
                    <td className="px-3 py-2 font-mono text-muted-foreground truncate max-w-0">
                      {typeof paper.source === 'string' ? paper.source : '—'}
                    </td>
                    <td className="px-3 py-2">
                      {/* 3-way control as pill tabs */}
                      <div
                        className="inline-flex h-7 items-center rounded-lg bg-muted p-0.5 gap-0.5"
                        role="group"
                        aria-label={`Decision for ${displayTitle}`}
                      >
                        {(['process', 'skip', 'discard'] as CandidateDecision[]).map((d) => (
                          <button
                            key={d}
                            onClick={() => setDecision(paper.id, d)}
                            aria-pressed={decision === d}
                            className={cn(
                              'px-2 py-0.5 rounded-md text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                              decision === d
                                ? d === 'process'
                                  ? 'bg-background text-blue-700 dark:text-blue-300 shadow-sm ring-1 ring-blue-200 dark:ring-blue-800'
                                  : d === 'discard'
                                    ? 'bg-background text-red-700 dark:text-red-300 shadow-sm ring-1 ring-red-200 dark:ring-red-800'
                                    : 'bg-background text-foreground shadow-sm'
                                : 'text-muted-foreground hover:text-foreground'
                            )}
                          >
                            {d === 'process' ? 'Process' : d === 'discard' ? 'Discard' : 'Skip'}
                          </button>
                        ))}
                      </div>
                    </td>
                    <td className="px-3 py-2">
                      <button
                        onClick={() => setEditingId(paper.id)}
                        className="p-1 rounded hover:bg-muted transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        aria-label={`Edit fields for ${displayTitle}`}
                        title="Edit fields"
                      >
                        <Pencil className="h-3.5 w-3.5 text-muted-foreground" />
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Repos (light listing) */}
      {repos.length > 0 && (
        <div>
          <p className="text-xs font-medium text-muted-foreground mb-2">
            Repo candidates ({repos.length}) — no action required, will be processed automatically
          </p>
          <div className="space-y-1">
            {repos.map((r) => (
              <div key={r.id} className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="font-mono">{r.owner}/{r.repo}</span>
                {r.description && <span className="truncate">— {r.description}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Submit */}
      <div className="flex items-center gap-3 pt-1">
        <Button
          onClick={handleSubmit}
          disabled={submitting || processIds.length === 0 && papers.length > 0}
          className="gap-1.5"
        >
          {submitting ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
          ) : (
            <Play className="h-3.5 w-3.5" aria-hidden="true" />
          )}
          Submit &amp; process {processIds.length}
        </Button>
        <p className="text-xs text-muted-foreground">
          {processIds.length === 0 && papers.length > 0
            ? 'Select at least one paper to process.'
            : null}
        </p>
      </div>

      {/* Edit dialog */}
      {editingPaper && (
        <EditDialog
          paper={editingPaper}
          open={editingId !== null}
          onClose={() => setEditingId(null)}
          onSave={handleEditSave}
          existing={edits[editingPaper.id] ?? {}}
        />
      )}
    </div>
  )
}

// ─── Live run view ────────────────────────────────────────────────────────────

interface LiveRunViewProps {
  runId: string
  mode: RunMode
  onDone: () => void
}

function LiveRunView({ runId, mode, onDone }: LiveRunViewProps) {
  const qc = useQueryClient()
  const eventState = useRunEvents(runId)
  const [cancelOpen, setCancelOpen] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [gateSubmitted, setGateSubmitted] = useState(false)

  const isRunning = !eventState.ended && eventState.derivedState !== 'awaiting_review'
  const isAwaiting = eventState.derivedState === 'awaiting_review' && !gateSubmitted
  const isDone =
    eventState.derivedState === 'done' ||
    eventState.derivedState === 'error' ||
    eventState.derivedState === 'cancelled'

  // Notify parent on completion
  useEffect(() => {
    if (isDone) {
      // Invalidate run queries so history refreshes
      qc.invalidateQueries({ queryKey: ['runs'] })
      qc.invalidateQueries({ queryKey: ['runs', 'active'] })
    }
  }, [isDone, qc])

  async function handleCancel() {
    setCancelling(true)
    try {
      await cancelRun(runId)
      toast.info('Cancel requested')
      setCancelOpen(false)
    } catch (err) {
      toast.error(`Failed to cancel: ${(err as Error).message}`)
    } finally {
      setCancelling(false)
    }
  }

  function handleGateSubmitted() {
    setGateSubmitted(true)
  }

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-3 justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-muted-foreground">{runId}</span>
          <span className={cn(runStateBadge(eventState.derivedState ?? 'running'))}>
            {runStateLabel(eventState.derivedState ?? 'running')}
          </span>
          <span className="text-xs text-muted-foreground capitalize">({mode})</span>
        </div>
        {isRunning && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => setCancelOpen(true)}
            className="gap-1.5 text-destructive hover:text-destructive border-destructive/30 hover:border-destructive/60"
          >
            <Square className="h-3.5 w-3.5" />
            Cancel
          </Button>
        )}
        {isDone && (
          <Button size="sm" variant="outline" onClick={onDone}>
            Start new run
          </Button>
        )}
      </div>

      {/* Rate limit cooldown */}
      {typeof eventState.rateLimitWait === 'number' && eventState.rateLimitWait > 0 && (
        <CooldownBadge seconds={eventState.rateLimitWait} label={eventState.rateLimitLabel} />
      )}

      {/* Phase stepper */}
      <PhaseStepper
        stageStatus={eventState.stageStatus}
        showReview={mode === 'incremental'}
      />

      {/* Per-item progress */}
      {eventState.currentItem && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin motion-reduce:animate-none flex-none text-blue-500" aria-hidden="true" />
          <span className="tabular-nums font-medium text-foreground">
            {eventState.currentItem.idx} / {eventState.currentItem.total}
          </span>
          <span className="truncate max-w-xs" title={eventState.currentItem.title}>
            {eventState.currentItem.title}
          </span>
        </div>
      )}

      {/* Review gate (incremental only, when awaiting) */}
      {isAwaiting && eventState.candidates && (
        <Card>
          <CardContent className="pt-4">
            <ReviewGate
              runId={runId}
              papers={eventState.candidates.papers}
              repos={eventState.candidates.repos}
              onSubmitted={handleGateSubmitted}
            />
          </CardContent>
        </Card>
      )}

      {/* Processing resumed after gate */}
      {gateSubmitted && eventState.derivedState !== 'awaiting_review' && (
        <div className="text-xs text-muted-foreground">Gate submitted — processing in progress…</div>
      )}

      {/* Error message */}
      {eventState.error && (
        <div className="flex items-center gap-2 text-destructive text-xs">
          <AlertCircle className="h-4 w-4 flex-none" aria-hidden="true" />
          {eventState.error}
        </div>
      )}

      {/* Log feed */}
      <div className="flex flex-col" style={{ minHeight: '200px', maxHeight: '340px' }}>
        <p className="text-xs font-medium text-muted-foreground mb-1.5">Event log</p>
        <EventLog events={eventState.events} />
      </div>

      {/* Cancel AlertDialog */}
      <AlertDialog open={cancelOpen} onOpenChange={setCancelOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel run?</AlertDialogTitle>
            <AlertDialogDescription>
              The pipeline will stop after the current item finishes. Partial results won't be saved.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setCancelOpen(false)} disabled={cancelling}>
              Keep running
            </AlertDialogCancel>
            <AlertDialogAction onClick={handleCancel} disabled={cancelling}>
              {cancelling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
              Yes, cancel
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

// ─── Trigger panel ────────────────────────────────────────────────────────────

interface TriggerPanelProps {
  onStarted: (runId: string, mode: RunMode) => void
}

function TriggerPanel({ onStarted }: TriggerPanelProps) {
  const [mode, setMode] = useState<RunMode>('incremental')
  const [skipPapers, setSkipPapers] = useState(false)
  const [skipGithub, setSkipGithub] = useState(false)
  const [starting, setStarting] = useState(false)

  // Preflight check — refetches when skip toggles change
  const { data: preflight, isLoading: preflightLoading, isError: preflightError } = useQuery({
    queryKey: ['preflight', skipPapers, skipGithub],
    queryFn: () => getPreflight(skipPapers, skipGithub, true),  // validate=true: catch a present-but-dead GitHub token
    staleTime: 30_000,
  })

  // Mutual exclusion: can't skip both
  function handleSkipPapers(v: boolean) {
    setSkipPapers(v)
    if (v) setSkipGithub(false)
  }
  function handleSkipGithub(v: boolean) {
    setSkipGithub(v)
    if (v) setSkipPapers(false)
  }

  async function handleStart() {
    setStarting(true)
    try {
      const result = await startRun({
        mode,
        skip_papers: skipPapers || undefined,
        skip_github: skipGithub || undefined,
      })
      onStarted(result.run_id, mode)
      toast.success('Run started')
    } catch (err) {
      const status = (err as { status?: number }).status
      if (status === 409) {
        toast.error('A run is already active — wait for it to finish first.')
      } else {
        toast.error(`Failed to start run: ${(err as Error).message}`)
      }
    } finally {
      setStarting(false)
    }
  }

  const preflightBlocked = !!preflight && !preflight.ok

  return (
    <div className="space-y-5">
      {/* Mode selector */}
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Mode</p>
        <div
          className="inline-flex h-8 items-center rounded-lg bg-muted p-1 gap-0.5"
          role="radiogroup"
          aria-label="Run mode"
        >
          {([
            { value: 'incremental', label: 'Incremental' },
            { value: 'fresh', label: 'Fresh' },
          ] as { value: RunMode; label: string }[]).map(({ value, label }) => (
            <button
              key={value}
              role="radio"
              aria-checked={mode === value}
              onClick={() => setMode(value)}
              className={cn(
                'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1 text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                mode === value
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {label}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground max-w-sm">
          {mode === 'incremental'
            ? 'Discovers new papers and pauses at a review gate before running LLM enrichment.'
            : 'Full rebuild — re-processes all papers from scratch. No review gate.'}
        </p>
      </div>

      {/* Skip toggles */}
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Skip stages</p>
        <div className="space-y-1.5">
          {[
            { id: 'skip-papers', label: 'Skip papers', value: skipPapers, onChange: handleSkipPapers },
            { id: 'skip-github', label: 'Skip GitHub repos', value: skipGithub, onChange: handleSkipGithub },
          ].map(({ id, label, value, onChange }) => (
            <label key={id} className="flex items-center gap-2 cursor-pointer select-none group">
              <span
                role="checkbox"
                aria-checked={value}
                tabIndex={0}
                id={id}
                onClick={() => onChange(!value)}
                onKeyDown={(e) => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); onChange(!value) } }}
                className={cn(
                  'w-4 h-4 rounded border flex items-center justify-center flex-none transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  value
                    ? 'bg-primary border-primary text-primary-foreground'
                    : 'border-input bg-background group-hover:border-ring'
                )}
                aria-labelledby={`${id}-label`}
              >
                {value && <Check className="h-3 w-3" aria-hidden="true" />}
              </span>
              <span id={`${id}-label`} className="text-xs text-foreground">{label}</span>
            </label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">Skip toggles are mutually exclusive.</p>
      </div>

      {/* Preflight blockers */}
      {preflight && preflight.blocking.length > 0 && (
        <div className="space-y-1">
          {preflight.blocking.map((msg, i) => (
            <div key={i} className="flex items-start gap-1.5 text-xs text-destructive">
              <AlertCircle className="h-3.5 w-3.5 flex-none mt-px" aria-hidden="true" />
              <span>{msg}</span>
            </div>
          ))}
        </div>
      )}

      {/* Preflight warnings */}
      {preflight && preflight.warnings.length > 0 && (
        <div className="space-y-1">
          {preflight.warnings.map((msg, i) => (
            <div key={i} className="flex items-start gap-1.5 text-xs text-amber-700 dark:text-amber-400">
              <AlertCircle className="h-3.5 w-3.5 flex-none mt-px" aria-hidden="true" />
              <span>{msg}</span>
            </div>
          ))}
        </div>
      )}

      {/* Preflight status (locking stays safe-default: a failed/loading check never blocks) */}
      {preflightLoading && (
        <p className="text-xs text-muted-foreground">Checking credentials…</p>
      )}
      {preflightError && (
        <p className="text-xs text-amber-700 dark:text-amber-400">Could not verify credentials — proceed with caution.</p>
      )}

      {/* Start button */}
      <Button onClick={handleStart} disabled={starting || preflightBlocked} className="gap-1.5">
        {starting ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
        ) : (
          <Play className="h-3.5 w-3.5" aria-hidden="true" />
        )}
        Start run
      </Button>
    </div>
  )
}

// ─── History list ─────────────────────────────────────────────────────────────

interface HistoryListProps {
  onSelectRun: (runId: string) => void
  activeRunId: string | null
}

function HistoryList({ onSelectRun, activeRunId }: HistoryListProps) {
  const { data: runs, isLoading } = useRuns()

  if (isLoading) {
    return (
      <div className="space-y-1.5">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-10 bg-muted animate-pulse rounded-md" />
        ))}
      </div>
    )
  }

  if (!runs || runs.length === 0) {
    return <p className="text-xs text-muted-foreground">No runs yet.</p>
  }

  return (
    <div className="space-y-0 rounded-md border overflow-hidden">
      {runs.map((run) => {
        const isActive = run.run_id === activeRunId
        return (
          <button
            key={run.run_id}
            onClick={() => onSelectRun(run.run_id)}
            className={cn(
              'w-full flex items-center gap-3 px-3 py-2.5 text-left text-xs border-b last:border-b-0 hover:bg-muted/50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset',
              isActive && 'bg-blue-50/60 dark:bg-blue-950/30'
            )}
          >
            <span className="font-mono text-muted-foreground truncate flex-1 min-w-0">{run.run_id}</span>
            <span className={runStateBadge(run.state)}>
              {runStateLabel(run.state)}
            </span>
            <span className="text-muted-foreground whitespace-nowrap">{fmtDate(run.started_at)}</span>
            {run.total != null && (
              <span className="text-muted-foreground whitespace-nowrap tabular-nums">
                {run.processed ?? 0}/{run.total}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}

// ─── Finished run read-only view ──────────────────────────────────────────────

interface FinishedRunViewProps {
  run: RunRecord
  onBack: () => void
}

function FinishedRunView({ run, onBack }: FinishedRunViewProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-muted-foreground">{run.run_id}</span>
          <span className={runStateBadge(run.state)}>{runStateLabel(run.state)}</span>
          <span className="text-xs text-muted-foreground capitalize">({run.mode})</span>
        </div>
        <Button size="sm" variant="ghost" onClick={onBack} className="gap-1.5">
          <X className="h-3.5 w-3.5" />
          Close
        </Button>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div>
          <p className="text-muted-foreground">Started</p>
          <p className="font-medium mt-0.5">{fmtDate(run.started_at)}</p>
        </div>
        <div>
          <p className="text-muted-foreground">Finished</p>
          <p className="font-medium mt-0.5">{fmtDate(run.finished_at)}</p>
        </div>
        <div>
          <p className="text-muted-foreground">Duration</p>
          <p className="font-medium mt-0.5">{fmtDuration(run.started_at, run.finished_at)}</p>
        </div>
        {run.counts && (
          <div>
            <p className="text-muted-foreground">Counts</p>
            <p className="font-medium mt-0.5 font-mono">{Object.entries(run.counts).map(([k, v]) => `${k}:${v}`).join(' ')}</p>
          </div>
        )}
      </div>
      {run.error && (
        <div className="flex items-center gap-2 text-destructive text-xs">
          <AlertCircle className="h-4 w-4 flex-none" />
          {run.error}
        </div>
      )}
      {Array.isArray(run.events) && run.events.length > 0 && (
        <div className="flex flex-col" style={{ minHeight: '160px', maxHeight: '300px' }}>
          <p className="text-xs font-medium text-muted-foreground mb-1.5">Event log (recorded)</p>
          <EventLog events={run.events} />
        </div>
      )}
    </div>
  )
}

// ─── Main Runs page ───────────────────────────────────────────────────────────

export function Runs() {
  const qc = useQueryClient()
  const { data: activeData } = useActiveRun()

  const activeRecord = activeData?.active ?? null

  // Active run state (run_id + mode)
  const [activeRun, setActiveRun] = useState<{ runId: string; mode: RunMode } | null>(null)
  // Selected finished run for read-only view
  const [selectedRun, setSelectedRun] = useState<RunRecord | null>(null)

  // On mount: if there's a live active/awaiting run, attach to it
  const resumedRef = useRef(false)
  useEffect(() => {
    if (!resumedRef.current && activeRecord && !activeRun) {
      resumedRef.current = true
      setActiveRun({ runId: activeRecord.run_id, mode: activeRecord.mode })
    }
  }, [activeRecord, activeRun])

  function handleStarted(runId: string, mode: RunMode) {
    setSelectedRun(null)
    setActiveRun({ runId, mode })
    // Refresh active run query
    qc.invalidateQueries({ queryKey: ['runs', 'active'] })
  }

  function handleDone() {
    setActiveRun(null)
    qc.invalidateQueries({ queryKey: ['runs'] })
    qc.invalidateQueries({ queryKey: ['runs', 'active'] })
  }

  async function handleSelectRun(runId: string) {
    // If this is the active run, switch to live view
    if (activeRun?.runId === runId) {
      setSelectedRun(null)
      return
    }
    // Otherwise fetch and show read-only
    try {
      const record = await fetchRun(runId)
      setSelectedRun(record)
    } catch {
      toast.error('Failed to load run details')
    }
  }

  const showLive = !!activeRun

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Page header */}
      <div>
        <h2 className="text-base font-semibold">Runs</h2>
        <p className="text-xs text-muted-foreground mt-0.5">
          Manage and monitor curation pipeline runs
        </p>
      </div>

      {/* Main panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            {showLive ? (
              <>
                <Loader2 className="h-4 w-4 text-blue-500 animate-spin motion-reduce:animate-none" aria-hidden="true" />
                Live run
              </>
            ) : (
              <>
                <Play className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                Start a run
              </>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {showLive && activeRun ? (
            <LiveRunView
              key={activeRun.runId}
              runId={activeRun.runId}
              mode={activeRun.mode}
              onDone={handleDone}
            />
          ) : (
            <TriggerPanel onStarted={handleStarted} />
          )}
        </CardContent>
      </Card>

      {/* Finished run detail (read-only) */}
      {selectedRun && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <SkipForward className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              Run details
            </CardTitle>
          </CardHeader>
          <CardContent>
            <FinishedRunView run={selectedRun} onBack={() => setSelectedRun(null)} />
          </CardContent>
        </Card>
      )}

      {/* History */}
      <div className="space-y-3">
        <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Run history</h3>
        <HistoryList
          onSelectRun={handleSelectRun}
          activeRunId={activeRun?.runId ?? null}
        />
      </div>
    </div>
  )
}
