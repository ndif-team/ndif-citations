import { ExternalLink, Copy, Check, AlertCircle, Pencil, X, Upload, RotateCcw, ChevronUp, ChevronDown, Trash2, Lock, ChevronLeft, ChevronRight, FileText, RefreshCw, Loader2 } from 'lucide-react'
import { useState, useRef, useCallback, useEffect } from 'react'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet'
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogAction,
  AlertDialogCancel,
} from '@/components/ui/alert-dialog'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { usePaper, useActiveRun } from '@/api/hooks'
import { useRunEvents } from '@/hooks/useRunEvents'
import { useQueryClient } from '@tanstack/react-query'
import { bucketBadge, confidenceBadge, categoryBadge, categoryLabel } from '@/lib/tokens'
import { EDITABLE_FIELDS, SELECT_NONE } from '@/lib/editable'
import type { EditableFieldMeta } from '@/lib/editable'
import { editPaper, setPaperBucket, uploadPaperImage, reextractThumbnail, paperPdfUrl, attachPdf, backfillEvidence, reprocessPaper } from '@/api/client'
import type { Bucket, ConfidenceBand, Category, PaperDetail } from '@/api/types'
import { toast } from 'sonner'

interface Props {
  paperId: string | null
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
  hasPrev?: boolean
  hasNext?: boolean
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ExternalLinkButton({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
    >
      <ExternalLink className="h-3 w-3" aria-hidden="true" />
      {label}
    </a>
  )
}

function CopyBibtex({ bibtex }: { bibtex: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(bibtex)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">BibTeX</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          className="h-6 px-2 text-xs gap-1"
          aria-label={copied ? 'Copied' : 'Copy BibTeX'}
        >
          {copied ? (
            <><Check className="h-3 w-3 text-green-500" />Copied</>
          ) : (
            <><Copy className="h-3 w-3" />Copy</>
          )}
        </Button>
      </div>
      <pre className="text-[10px] leading-relaxed font-mono bg-muted p-2.5 rounded-md overflow-x-auto text-muted-foreground whitespace-pre-wrap break-all">
        {bibtex}
      </pre>
    </div>
  )
}

// ---------------------------------------------------------------------------
// EditForm: renders 16 editable fields pre-filled from paper data
// ---------------------------------------------------------------------------

/** Fields whose cleared state is represented by SELECT_NONE in the UI. */
const SELECT_NONE_FIELDS = new Set(['reason', 'peer_reviewed'])

/**
 * Read a field value from the paper for use as a Select/Input value.
 *
 * For clearable select fields (`reason`, `peer_reviewed`) an absent / null /
 * empty-string value is mapped to SELECT_NONE because Radix UI forbids
 * `value=""` on a `<SelectItem>`.
 */
function fieldValue(paper: PaperDetail, fieldName: string): string {
  const raw = (paper as unknown as Record<string, unknown>)[fieldName]
  let str: string
  if (raw === null || raw === undefined) {
    str = ''
  } else if (typeof raw === 'boolean') {
    str = raw ? 'yes' : 'no'
  } else {
    str = String(raw)
  }
  // Map empty string to sentinel for clearable select fields
  if (SELECT_NONE_FIELDS.has(fieldName) && str === '') {
    return SELECT_NONE
  }
  return str
}

interface EditFormProps {
  paper: PaperDetail
  disabled: boolean
  onSave: (changed: Record<string, string>) => void
  onCancel: () => void
  saving: boolean
}

function EditForm({ paper, disabled, onSave, onCancel, saving }: EditFormProps) {
  const [values, setValues] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {}
    for (const f of EDITABLE_FIELDS) {
      init[f.name] = fieldValue(paper, f.name)
    }
    return init
  })
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({})

  const original: Record<string, string> = {}
  for (const f of EDITABLE_FIELDS) {
    original[f.name] = fieldValue(paper, f.name)
  }

  function handleChange(name: string, value: string) {
    setValues(prev => ({ ...prev, [name]: value }))
    setFieldErrors(prev => {
      const next = { ...prev }
      delete next[name]
      return next
    })
  }

  function handleSave() {
    const changed: Record<string, string> = {}
    for (const f of EDITABLE_FIELDS) {
      if (values[f.name] !== original[f.name]) {
        // Map the sentinel back to "" so the backend receives the clear signal
        // (_parse_reason("") → None, _parse_bool("") → None)
        changed[f.name] = values[f.name] === SELECT_NONE ? '' : values[f.name]
      }
    }
    if (Object.keys(changed).length === 0) {
      onCancel()
      return
    }
    onSave(changed)
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-3">
        {EDITABLE_FIELDS.map(f => (
          <EditFieldRow
            key={f.name}
            field={f}
            value={values[f.name]}
            error={fieldErrors[f.name]}
            disabled={disabled || saving}
            onChange={(v) => handleChange(f.name, v)}
          />
        ))}
      </div>
      <div className="flex gap-2 pt-1">
        <Button size="sm" onClick={handleSave} disabled={disabled || saving}>
          {saving ? 'Saving…' : 'Save'}
        </Button>
        <Button size="sm" variant="outline" onClick={onCancel} disabled={saving}>
          Cancel
        </Button>
      </div>
    </div>
  )
}

interface EditFieldRowProps {
  field: EditableFieldMeta
  value: string
  error?: string
  disabled: boolean
  onChange: (v: string) => void
}

function EditFieldRow({ field, value, error, disabled, onChange }: EditFieldRowProps) {
  return (
    <div className="space-y-0.5">
      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        {field.label}
      </label>
      {field.type === 'select' && field.options ? (
        <Select value={value} onValueChange={onChange} disabled={disabled}>
          <SelectTrigger className="h-7 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {field.options.map(opt => (
              <SelectItem key={opt.value} value={opt.value} className="text-xs">
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : field.type === 'textarea' ? (
        <textarea
          value={value}
          onChange={e => onChange(e.target.value)}
          disabled={disabled}
          placeholder={field.placeholder}
          rows={3}
          className="flex w-full rounded-md border border-input bg-background px-3 py-1.5 text-xs shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 resize-y"
        />
      ) : (
        <Input
          type={field.type === 'number' ? 'number' : 'text'}
          value={value}
          onChange={e => onChange(e.target.value)}
          disabled={disabled}
          placeholder={field.placeholder}
          className="h-7 text-xs"
        />
      )}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// BucketActions: Promote / Demote (with reason) / Discard (with confirm)
// ---------------------------------------------------------------------------

interface BucketActionsProps {
  paper: PaperDetail
  paperId: string
  disabled: boolean
  onMutate: (updater: (prev: PaperDetail) => PaperDetail) => void
}

// Local reason options for the demote select — uses SELECT_NONE instead of ""
// to satisfy Radix UI's requirement that SelectItem values are non-empty.
const BUCKET_REASON_OPTIONS = [
  { value: SELECT_NONE,                label: '(none)' },
  { value: 'openalex_source',          label: 'OpenAlex source' },
  { value: 'low_confidence',           label: 'Low confidence' },
  { value: 'medium_confidence',        label: 'Medium confidence' },
  { value: 'unclassified_no_keywords', label: 'Unclassified — no keywords' },
  { value: 'unclassified_llm',         label: 'Unclassified — LLM' },
  { value: 'stub_metadata',            label: 'Stub metadata' },
  { value: 'zero_pdf_hits',            label: 'Zero PDF hits' },
  { value: 'manual_discard',           label: 'Manual discard' },
  { value: 'manual_demote',            label: 'Manual demote' },
]

function BucketActions({ paper, paperId, disabled, onMutate }: BucketActionsProps) {
  const [demoteReason, setDemoteReason] = useState(SELECT_NONE)
  const [discardOpen, setDiscardOpen] = useState(false)
  const [demoteOpen, setDemoteOpen] = useState(false)
  const [busy, setBusy] = useState(false)

  async function handleBucket(
    bucket: Bucket,
    reason?: string,
    detail?: string,
  ) {
    setBusy(true)
    const prevBucket = paper.bucket
    const prevReason = paper.reason ?? undefined
    try {
      const updated = await setPaperBucket(paperId, {
        bucket,
        reason: reason || undefined,
        detail,
      })
      onMutate(() => updated)
      toast.success(`Moved to ${bucket}`, {
        action: {
          label: 'Undo',
          onClick: async () => {
            try {
              const reverted = await setPaperBucket(paperId, {
                bucket: prevBucket,
                reason: prevReason,
              })
              onMutate(() => reverted)
              toast.success(`Reverted to ${prevBucket}`)
            } catch (err) {
              const msg = (err as { status?: number }).status === 409
                ? 'A run is in progress — try again when it finishes'
                : (err as Error).message
              toast.error(msg)
            }
          },
        },
      })
    } catch (err) {
      const msg = (err as { status?: number }).status === 409
        ? 'A run is in progress — try again when it finishes'
        : (err as Error).message
      toast.error(msg)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
        Bucket actions
      </p>
      <div className="flex flex-wrap gap-2">
        {/* Promote */}
        {paper.bucket !== 'verified' && (
          <Button
            size="sm"
            variant="outline"
            disabled={disabled || busy}
            onClick={() => handleBucket('verified')}
            className="h-7 text-xs gap-1"
          >
            <ChevronUp className="h-3 w-3" />
            Promote
          </Button>
        )}

        {/* Demote — opens inline reason selector */}
        {paper.bucket !== 'pending' && (
          <>
            {demoteOpen ? (
              <div className="flex items-center gap-1.5 flex-wrap">
                <Select value={demoteReason} onValueChange={setDemoteReason}>
                  <SelectTrigger className="h-7 text-xs w-48">
                    <SelectValue placeholder="Reason…" />
                  </SelectTrigger>
                  <SelectContent>
                    {BUCKET_REASON_OPTIONS.map(opt => (
                      <SelectItem key={opt.value} value={opt.value} className="text-xs">
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={disabled || busy}
                  onClick={async () => {
                    // Map sentinel back to undefined (no reason) before calling API
                    const reason = demoteReason === SELECT_NONE ? undefined : demoteReason
                    await handleBucket('pending', reason)
                    setDemoteOpen(false)
                    setDemoteReason(SELECT_NONE)
                  }}
                  className="h-7 text-xs"
                >
                  Confirm
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => { setDemoteOpen(false); setDemoteReason(SELECT_NONE) }}
                  className="h-7 text-xs"
                >
                  Cancel
                </Button>
              </div>
            ) : (
              <Button
                size="sm"
                variant="outline"
                disabled={disabled || busy}
                onClick={() => setDemoteOpen(true)}
                className="h-7 text-xs gap-1"
              >
                <ChevronDown className="h-3 w-3" />
                Demote
              </Button>
            )}
          </>
        )}

        {/* Discard — with AlertDialog confirm */}
        {paper.bucket !== 'discarded' && (
          <>
            <Button
              size="sm"
              variant="outline"
              disabled={disabled || busy}
              onClick={() => setDiscardOpen(true)}
              className="h-7 text-xs gap-1 text-destructive hover:text-destructive border-destructive/30 hover:border-destructive/60"
            >
              <Trash2 className="h-3 w-3" />
              Discard
            </Button>
            <AlertDialog open={discardOpen} onOpenChange={setDiscardOpen}>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Discard this paper?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will move the paper to the discarded bucket. You can undo via the toast.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel onClick={() => setDiscardOpen(false)}>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={async () => {
                      setDiscardOpen(false)
                      await handleBucket('discarded', 'manual_discard')
                    }}
                  >
                    Discard
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// ImageActions: Replace image (file upload) + Re-extract
// ---------------------------------------------------------------------------

interface ImageActionsProps {
  paperId: string
  disabled: boolean
  procRunning: boolean
  onMutate: (updater: (prev: PaperDetail) => PaperDetail) => void
  onRunStarted: (runId: string, label: string) => void
}

function ImageActions({ paperId, disabled, procRunning, onMutate, onRunStarted }: ImageActionsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (fileInputRef.current) fileInputRef.current.value = ''
    setUploading(true)
    try {
      const updated = await uploadPaperImage(paperId, file)
      onMutate(() => updated)
      toast.success('Image replaced')
    } catch (err) {
      const msg = (err as { status?: number }).status === 409
        ? 'A run is in progress — try again when it finishes'
        : (err as Error).message
      toast.error(msg)
    } finally {
      setUploading(false)
    }
  }, [paperId, onMutate])

  const handleReextract = useCallback(async () => {
    try {
      const { run_id } = await reextractThumbnail(paperId)
      onRunStarted(run_id, 'Thumbnail re-extract')
    } catch (err) {
      const msg = (err as { status?: number }).status === 409
        ? 'A run is in progress — try again when it finishes'
        : (err as Error).message
      toast.error(msg)
    }
  }, [paperId, onRunStarted])

  return (
    <div>
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
        Image
      </p>
      <div className="flex flex-wrap gap-2">
        <Button
          size="sm"
          variant="outline"
          disabled={disabled || uploading}
          onClick={() => fileInputRef.current?.click()}
          className="h-7 text-xs gap-1"
        >
          <Upload className="h-3 w-3" />
          {uploading ? 'Uploading…' : 'Replace image'}
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png"
          className="hidden"
          onChange={handleUpload}
          aria-hidden="true"
        />
        <Button
          size="sm"
          variant="outline"
          disabled={disabled || procRunning}
          onClick={handleReextract}
          className="h-7 text-xs gap-1"
        >
          <RotateCcw className="h-3 w-3" />
          Re-extract
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main PaperSheet
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Needs-attention strip helpers
// ---------------------------------------------------------------------------

const MISSING_FIELD_LABELS: Record<string, string> = {
  image: 'Thumbnail',
  affiliations: 'Affiliations',
  abstract: 'Abstract',
  summary: 'Summary',
  venue: 'Venue',
}

// Fields that can be fixed by entering edit mode
const EDITABLE_MISSING_FIELDS = new Set(['venue', 'affiliations', 'summary', 'abstract'])

export function PaperSheet({ paperId, onClose, onPrev, onNext, hasPrev, hasNext }: Props) {
  const qc = useQueryClient()
  const { data: paper, isLoading, error } = usePaper(paperId)
  const { data: activeRunData } = useActiveRun()
  const hasActiveRun = !!(activeRunData?.active)

  const [editMode, setEditMode] = useState(false)
  const [saving, setSaving] = useState(false)
  const [lightbox, setLightbox] = useState(false)
  const [postAttachOpen, setPostAttachOpen] = useState(false)
  const [confirmField, setConfirmField] = useState<null | 'summary' | 'classify'>(null)
  const [processingBusy, setProcessingBusy] = useState(false)
  const pdfInputRef = useRef<HTMLInputElement>(null)
  const lightboxRef = useRef<HTMLDivElement>(null)

  // Async background job subscription (thumbnail re-extract + reprocess)
  const [procRunId, setProcRunId] = useState<string | null>(null)
  const [procLabel, setProcLabel] = useState('')
  const procEvents = useRunEvents(procRunId)

  const handleRunStarted = useCallback((runId: string, label: string) => {
    setProcLabel(label)
    setProcRunId(runId)
  }, [])

  useEffect(() => {
    if (procRunId && procEvents.ended) {
      qc.invalidateQueries({ queryKey: ['paper', paperId] })
      qc.invalidateQueries({ queryKey: ['papers'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      toast.success(`${procLabel} done`)
      setProcRunId(null)
    }
  }, [procRunId, procEvents.ended, qc, paperId, procLabel])

  const isOpen = !!paperId

  async function runReprocess(field: 'summary' | 'classify') {
    if (!paperId) return
    setProcessingBusy(true)
    try {
      const { run_id } = await reprocessPaper(paperId, [field])
      handleRunStarted(run_id, field === 'summary' ? 'Summarize' : 'Categorize')
    } catch (e) {
      const status = (e as { status?: number }).status
      toast.error(status === 409 ? 'A run is already active — wait for it to finish.' : (e as Error).message || 'Reprocess failed')
    } finally {
      setConfirmField(null)
      setProcessingBusy(false)
    }
  }

  // Lightbox: Escape to close + focus trap. The parent Radix Sheet is told to
  // ignore Escape while the lightbox is open (see onEscapeKeyDown on SheetContent
  // below), so this handler closes only the lightbox — not the whole panel. On
  // open we move focus into the dialog; Tab/Shift+Tab cycle within it; on close
  // focus is restored to the trigger (deferred a frame so it wins any Radix
  // focus-scope re-assertion after the dialog node is removed).
  useEffect(() => {
    if (!lightbox) return
    const trigger = document.activeElement as HTMLElement | null
    const node = lightboxRef.current
    node?.focus()
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setLightbox(false)
        return
      }
      if (e.key !== 'Tab' || !node) return
      const focusable = Array.from(
        node.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      )
      e.preventDefault()
      if (focusable.length === 0) {
        node.focus()
        return
      }
      const idx = focusable.indexOf(document.activeElement as HTMLElement)
      const next = e.shiftKey
        ? (idx <= 0 ? focusable[focusable.length - 1] : focusable[idx - 1])
        : (idx === -1 || idx === focusable.length - 1 ? focusable[0] : focusable[idx + 1])
      next.focus()
    }
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('keydown', handleKey)
      requestAnimationFrame(() => {
        if (trigger && trigger.isConnected) trigger.focus()
      })
    }
  }, [lightbox])

  // Arrow key navigation (while sheet is open and not in edit mode)
  useEffect(() => {
    if (!isOpen) return
    function handleNavKey(e: KeyboardEvent) {
      // Skip if user is typing in a form field or content-editable
      const target = e.target as HTMLElement
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target.isContentEditable
      ) return
      // Skip if edit form is open
      if (editMode) return
      // Skip if lightbox overlay is open
      if (lightbox) return
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        onPrev?.()
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        onNext?.()
      }
    }
    document.addEventListener('keydown', handleNavKey)
    return () => document.removeEventListener('keydown', handleNavKey)
  }, [isOpen, editMode, lightbox, onPrev, onNext])

  // When paperId changes (new paper selected), exit edit mode
  const prevPaperId = useRef<string | null>(null)
  if (paperId !== prevPaperId.current) {
    prevPaperId.current = paperId
    // Defer state reset to avoid setState during render
  }

  /** Optimistically update the paper in TanStack Query cache. */
  const handleMutate = useCallback(
    (updater: (prev: PaperDetail) => PaperDetail) => {
      if (!paperId) return
      qc.setQueryData<PaperDetail>(['paper', paperId], old => {
        if (!old) return old
        return updater(old)
      })
      // Also invalidate the papers list so the row updates
      qc.invalidateQueries({ queryKey: ['papers'] })
    },
    [paperId, qc],
  )

  const handleSave = useCallback(
    async (changed: Record<string, string>) => {
      if (!paper || !paperId) return
      setSaving(true)
      try {
        const updated = await editPaper(paperId, changed)
        handleMutate(() => updated)
        setEditMode(false)
        toast.success('Saved')
      } catch (err) {
        const status = (err as { status?: number }).status
        if (status === 409) {
          toast.error('A run is in progress — try again when it finishes')
        } else if (status === 422) {
          // Try to parse field-level detail from FastAPI 422
          const msg = (err as Error).message
          toast.error(`Validation error: ${msg}`)
        } else {
          toast.error((err as Error).message)
        }
      } finally {
        setSaving(false)
      }
    },
    [paper, paperId, handleMutate],
  )

  async function handlePdfFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file || !paperId) return
    try {
      const updated = await attachPdf(paperId, file)
      handleMutate(() => updated)
      setPostAttachOpen(true)
      toast.success('PDF attached')
    } catch (err) {
      toast.error((err as Error).message || 'Attach failed')
    }
  }

  return (
    <Sheet open={isOpen} onOpenChange={(open) => {
      if (!open) {
        setEditMode(false)
        onClose()
      }
    }}>
      <SheetContent
        side="right"
        className="overflow-y-auto p-0 flex flex-col w-[420px] sm:max-w-[420px]"
        onEscapeKeyDown={(e) => { if (lightbox) e.preventDefault() }}
      >
        {/* Radix requires an accessible title/description on every open dialog. The
            loaded branch renders its own visible SheetTitle; during loading/error
            (and the first paint before the query resolves) supply a hidden one so
            the Sheet is never title-less (F-010). Guarded on !paper to avoid two
            titles (and duplicate ids) once the paper is loaded. */}
        {!paper && (
          <>
            <SheetTitle className="sr-only">Paper details</SheetTitle>
            <SheetDescription className="sr-only">Loading paper details.</SheetDescription>
          </>
        )}

        {isLoading && (
          <div className="p-6 space-y-4">
            <Skeleton className="h-5 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
            <Skeleton className="h-40 w-full rounded-md" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-4/6" />
          </div>
        )}

        {error && (
          <div className="p-6 flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="h-4 w-4 flex-none" aria-hidden="true" />
            <span>Failed to load paper: {(error as Error).message}</span>
          </div>
        )}

        {paper && (
          <>
            {/* Lightbox overlay */}
            {lightbox && paper.image && (
              <div
                ref={lightboxRef}
                tabIndex={-1}
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6 outline-none"
                onClick={() => setLightbox(false)}
                role="dialog"
                aria-modal="true"
                aria-label={`Thumbnail preview — ${paper.title}`}
              >
                <button
                  type="button"
                  className="absolute right-4 top-4 rounded-md bg-white/10 p-2 text-white hover:bg-white/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
                  onClick={(e) => { e.stopPropagation(); setLightbox(false) }}
                  aria-label="Close preview"
                >
                  <X className="h-5 w-5" />
                </button>
                <img
                  src={`/api/images/${encodeURIComponent(paper.image.split('/').pop() ?? '')}`}
                  alt={`Thumbnail for ${paper.title}`}
                  className="max-h-full max-w-full rounded shadow-2xl"
                />
              </div>
            )}

            {/* Thumbnail */}
            {paper.image ? (
              <div className="p-4 bg-muted/30 border-b">
                <button
                  type="button"
                  className="w-full cursor-zoom-in focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md"
                  onClick={() => setLightbox(true)}
                  aria-label="Zoom thumbnail"
                >
                  <img
                    src={`/api/images/${encodeURIComponent(paper.image.split('/').pop() ?? '')}`}
                    alt={`Thumbnail for ${paper.title}`}
                    className="w-full max-h-48 object-contain rounded-md"
                    loading="lazy"
                  />
                </button>
              </div>
            ) : (
              <div className="p-4 border-b flex items-center justify-center h-24 bg-muted/20">
                <span className="text-xs text-muted-foreground">No thumbnail</span>
              </div>
            )}

            <SheetHeader className="px-5 pt-5 pb-3">
              <div className="flex items-start justify-between gap-2 pr-6">
                <SheetTitle className="text-sm font-semibold leading-snug flex-1">
                  {paper.title}
                </SheetTitle>
                <div className="flex items-center gap-0.5 flex-none mt-0.5">
                  {/* Prev / Next navigation */}
                  {(onPrev || onNext) && (
                    <>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={onPrev}
                        disabled={!hasPrev}
                        aria-label="Previous paper"
                        title="Previous paper (←)"
                      >
                        <ChevronLeft className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={onNext}
                        disabled={!hasNext}
                        aria-label="Next paper"
                        title="Next paper (→)"
                      >
                        <ChevronRight className="h-3.5 w-3.5" />
                      </Button>
                    </>
                  )}
                  {/* Edit toggle */}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => setEditMode(e => !e)}
                    aria-label={editMode ? 'Cancel editing' : 'Edit paper'}
                    title={editMode ? 'Cancel editing' : 'Edit paper'}
                  >
                    {editMode ? <X className="h-3.5 w-3.5" /> : <Pencil className="h-3.5 w-3.5" />}
                  </Button>
                </div>
              </div>
              {paper.authors && paper.authors.trim().length > 0 && (
                <SheetDescription className="text-xs">
                  {paper.authors}
                </SheetDescription>
              )}
            </SheetHeader>

            {/* Needs-attention strip */}
            {(paper.missing ?? []).length > 0 && (
              <div className="px-5 pb-2 flex flex-wrap gap-1.5">
                {(paper.missing ?? []).map(field => {
                  const label = MISSING_FIELD_LABELS[field] ?? field
                  const isEditable = EDITABLE_MISSING_FIELDS.has(field)
                  return (
                    <button
                      key={field}
                      type="button"
                      onClick={isEditable ? () => setEditMode(true) : undefined}
                      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium ring-1 ring-inset bg-amber-50 text-amber-700 ring-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-800 ${isEditable ? 'cursor-pointer hover:bg-amber-100 dark:hover:bg-amber-900' : 'cursor-default'}`}
                      aria-label={isEditable ? `Missing ${label} — click to edit` : `Missing ${label}`}
                    >
                      {label}
                    </button>
                  )
                })}
              </div>
            )}

            <div className="px-5 pb-5 space-y-4 flex-1">
              {editMode ? (
                /* ---- Edit form ---- */
                <EditForm
                  paper={paper}
                  disabled={hasActiveRun}
                  saving={saving}
                  onSave={handleSave}
                  onCancel={() => setEditMode(false)}
                />
              ) : (
                /* ---- Read view ---- */
                <>
                  {/* Badges row */}
                  <div className="flex flex-wrap gap-1.5">
                    <span className={bucketBadge(paper.bucket as Bucket)}>
                      {paper.bucket}
                    </span>
                    <span className={categoryBadge(paper.category as Category)}>
                      {categoryLabel(paper.category as Category)}
                    </span>
                    {paper.confidence_band && paper.confidence_band !== 'NONE' && (
                      <span className={confidenceBadge(paper.confidence_band as ConfidenceBand)}>
                        {paper.confidence_band}
                      </span>
                    )}
                    {paper.manual_override && (
                      <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset bg-violet-50 text-violet-800 ring-violet-200 dark:bg-violet-950 dark:text-violet-300 dark:ring-violet-800">
                        <Lock className="h-2.5 w-2.5" aria-hidden="true" />
                        Curator-locked
                      </span>
                    )}
                  </div>

                  {/* Venue & year */}
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                    {paper.venue && (
                      <span>
                        <span className="font-medium text-foreground">{paper.venue}</span>
                        {paper.venue_source && (
                          <span className="ml-1 opacity-60">({paper.venue_source})</span>
                        )}
                      </span>
                    )}
                    {paper.year && (
                      <span className="tabular-nums">{paper.year}</span>
                    )}
                  </div>

                  {/* Affiliations */}
                  {paper.affiliations && paper.affiliations.trim().length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Affiliations</p>
                      <p className="text-xs text-muted-foreground">{paper.affiliations}</p>
                    </div>
                  )}

                  {/* Reason */}
                  {paper.reason && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Classification reason</p>
                      <p className="text-xs text-foreground">{paper.reason}</p>
                      {paper.reason_detail && (
                        <p className="text-xs text-muted-foreground mt-0.5">{paper.reason_detail}</p>
                      )}
                    </div>
                  )}

                  {/* Abstract */}
                  <div>
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Abstract</p>
                    {paper.abstract ? (
                      <div className="max-h-64 overflow-y-auto pr-1">
                        <p className="text-xs leading-relaxed text-foreground">{paper.abstract}</p>
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground italic">No abstract available.</p>
                    )}
                  </div>

                  {/* Links */}
                  {(paper.url || paper.pdf_url || paper.project_url || paper.has_pdf) && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Links</p>
                      <div className="flex flex-wrap gap-3">
                        {paper.url && <ExternalLinkButton href={paper.url} label="Paper" />}
                        {paper.pdf_url && <ExternalLinkButton href={paper.pdf_url} label="PDF" />}
                        {paper.project_url && <ExternalLinkButton href={paper.project_url} label="Project" />}
                        {paper.has_pdf && <ExternalLinkButton href={paperPdfUrl(paperId ?? '')} label="Cached PDF" />}
                      </div>
                    </div>
                  )}

                  {/* PDF attach/replace */}
                  <div>
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">PDF</p>
                    <input
                      ref={pdfInputRef}
                      type="file"
                      accept="application/pdf"
                      className="hidden"
                      onChange={handlePdfFile}
                      aria-hidden="true"
                    />
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 text-xs gap-1"
                      disabled={hasActiveRun}
                      title={hasActiveRun ? 'A run is already active' : undefined}
                      onClick={() => pdfInputRef.current?.click()}
                    >
                      <FileText className="h-3 w-3" />
                      {paper.has_pdf ? 'Replace PDF' : 'Attach PDF'}
                    </Button>
                  </div>

                  {/* Post-attach backfill offer */}
                  <AlertDialog open={postAttachOpen} onOpenChange={setPostAttachOpen}>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>PDF attached — run a backfill?</AlertDialogTitle>
                        <AlertDialogDescription>
                          Would you like to run a no-LLM backfill using the new PDF?
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter className="flex-col sm:flex-row gap-2">
                        <Button variant="ghost" size="sm" onClick={() => setPostAttachOpen(false)}>
                          Not now
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={hasActiveRun}
                          title={hasActiveRun ? 'A run is already active' : undefined}
                          onClick={async () => {
                            setPostAttachOpen(false)
                            try {
                              const { run_id } = await reextractThumbnail(paperId ?? '')
                              handleRunStarted(run_id, 'Thumbnail re-extract')
                            } catch (err) {
                              toast.error((err as Error).message || 'Re-extract failed')
                            }
                          }}
                        >
                          Re-extract thumbnail
                        </Button>
                        <Button
                          size="sm"
                          disabled={hasActiveRun}
                          title={hasActiveRun ? 'A run is already active' : undefined}
                          onClick={async () => {
                            setPostAttachOpen(false)
                            try {
                              const u = await backfillEvidence(paperId ?? '')
                              handleMutate(() => u)
                              toast.success('Evidence backfilled')
                            } catch (err) {
                              toast.error((err as Error).message || 'Backfill failed')
                            }
                          }}
                        >
                          Backfill evidence
                        </Button>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>

                  {/* Evidence */}
                  <details className="group">
                    <summary className="flex items-center gap-2 cursor-pointer list-none text-xs font-medium text-muted-foreground uppercase tracking-wider select-none">
                      <span>Evidence</span>
                      {paper.context_source === 'pdf' ? (
                        <span className="inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium ring-1 ring-inset bg-blue-50 text-blue-700 ring-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:ring-blue-800">
                          from PDF
                        </span>
                      ) : paper.context_source === 'abstract' ? (
                        <span className="inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium ring-1 ring-inset bg-green-50 text-green-700 ring-green-200 dark:bg-green-950 dark:text-green-300 dark:ring-green-800">
                          from abstract
                        </span>
                      ) : (
                        <span className="inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium ring-1 ring-inset bg-muted text-muted-foreground ring-border">
                          no evidence
                        </span>
                      )}
                    </summary>
                    <div className="mt-2 space-y-2">
                      {(paper.ndif_context_windows ?? []).length > 0 ? (
                        (paper.ndif_context_windows ?? []).map((window, i) => (
                          <p
                            key={i}
                            className="font-mono text-xs whitespace-pre-wrap bg-muted/60 text-muted-foreground rounded-md p-2.5 border border-border/50"
                          >
                            {window}
                          </p>
                        ))
                      ) : (
                        <p className="text-xs text-muted-foreground italic">No NDIF evidence found in the source.</p>
                      )}
                    </div>
                  </details>

                  {/* Bucket actions */}
                  <BucketActions
                    paper={paper}
                    paperId={paperId ?? ''}
                    disabled={hasActiveRun}
                    onMutate={handleMutate}
                  />

                  {/* Image management */}
                  <ImageActions
                    paperId={paperId ?? ''}
                    disabled={hasActiveRun}
                    procRunning={!!procRunId && !procEvents.ended}
                    onMutate={handleMutate}
                    onRunStarted={handleRunStarted}
                  />

                  {/* Processing */}
                  <div>
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Processing</p>
                    <div className="flex flex-wrap gap-2 items-center">
                      <Button size="sm" variant="outline" className="h-7 text-xs gap-1"
                              disabled={hasActiveRun || processingBusy || !!procRunId}
                              title={hasActiveRun ? 'A run is already active' : undefined}
                              onClick={() => setConfirmField('summary')}>
                        <RefreshCw className="h-3 w-3" /> Summarize
                      </Button>
                      <Button size="sm" variant="outline" className="h-7 text-xs gap-1"
                              disabled={hasActiveRun || processingBusy || !!procRunId}
                              title={hasActiveRun ? 'A run is already active' : undefined}
                              onClick={() => setConfirmField('classify')}>
                        <RefreshCw className="h-3 w-3" /> Categorize
                      </Button>
                      {procRunId && !procEvents.ended && (
                        <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                          <Loader2 className="h-3.5 w-3.5 animate-spin" /> {procLabel}…
                        </span>
                      )}
                    </div>
                  </div>

                  <AlertDialog open={confirmField !== null} onOpenChange={(o) => !o && setConfirmField(null)}>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>
                          {confirmField === 'summary' ? 'Re-summarize this paper?' : 'Re-categorize this paper?'}
                        </AlertDialogTitle>
                        <AlertDialogDescription>
                          This runs the LLM (spends an API call) on this paper. Continue?
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel onClick={() => setConfirmField(null)}>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          className="bg-primary text-primary-foreground hover:bg-primary/90"
                          onClick={() => confirmField && runReprocess(confirmField)}
                        >
                          {confirmField === 'summary' ? 'Summarize' : 'Categorize'}
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>

                  {/* ID */}
                  {(() => {
                    const idLabel = paper.arxiv_id ? 'arXiv' : paper.doi ? 'DOI' : 'ID'
                    const idValue = paper.arxiv_id ?? paper.doi ?? paperId ?? ''
                    return idValue ? (
                      <div>
                        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">{idLabel}</p>
                        <code className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded break-all">
                          {idValue}
                        </code>
                      </div>
                    ) : null
                  })()}

                  {/* BibTeX */}
                  {paper.bibtex && <CopyBibtex bibtex={paper.bibtex} />}
                </>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
