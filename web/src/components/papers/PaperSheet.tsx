import { ExternalLink, Copy, Check, AlertCircle, Pencil, X, Upload, RotateCcw, ChevronUp, ChevronDown, Trash2 } from 'lucide-react'
import { useState, useRef, useCallback } from 'react'
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
import { useQueryClient } from '@tanstack/react-query'
import { bucketBadge, confidenceBadge, categoryBadge, categoryLabel } from '@/lib/tokens'
import { EDITABLE_FIELDS, SELECT_NONE } from '@/lib/editable'
import type { EditableFieldMeta } from '@/lib/editable'
import { editPaper, setPaperBucket, uploadPaperImage, reextractThumbnail } from '@/api/client'
import type { Bucket, ConfidenceBand, Category, PaperDetail } from '@/api/types'
import { toast } from 'sonner'

interface Props {
  paperId: string | null
  onClose: () => void
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

function BucketActions({ paper, disabled, onMutate }: BucketActionsProps) {
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
      const updated = await setPaperBucket(paper.id, {
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
              const reverted = await setPaperBucket(paper.id, {
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
  paper: PaperDetail
  disabled: boolean
  onMutate: (updater: (prev: PaperDetail) => PaperDetail) => void
}

function ImageActions({ paper, disabled, onMutate }: ImageActionsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (fileInputRef.current) fileInputRef.current.value = ''
    setUploading(true)
    try {
      const updated = await uploadPaperImage(paper.id, file)
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
  }, [paper.id, onMutate])

  const handleReextract = useCallback(async () => {
    try {
      await reextractThumbnail(paper.id)
      toast.info('Re-extracting thumbnail… check the run indicator for progress')
    } catch (err) {
      const msg = (err as { status?: number }).status === 409
        ? 'A run is in progress — try again when it finishes'
        : (err as Error).message
      toast.error(msg)
    }
  }, [paper.id])

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
          disabled={disabled}
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

export function PaperSheet({ paperId, onClose }: Props) {
  const qc = useQueryClient()
  const { data: paper, isLoading, error } = usePaper(paperId)
  const { data: activeRunData } = useActiveRun()
  const hasActiveRun = !!(activeRunData?.active)

  const [editMode, setEditMode] = useState(false)
  const [saving, setSaving] = useState(false)

  const isOpen = !!paperId

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
      if (!paper) return
      setSaving(true)
      try {
        const updated = await editPaper(paper.id, changed)
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
    [paper, handleMutate],
  )

  return (
    <Sheet open={isOpen} onOpenChange={(open) => {
      if (!open) {
        setEditMode(false)
        onClose()
      }
    }}>
      <SheetContent side="right" className="overflow-y-auto p-0 flex flex-col w-[420px] sm:max-w-[420px]">
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
            {/* Thumbnail */}
            {paper.has_image && paper.image && (
              <div className="p-4 bg-muted/30 border-b">
                <img
                  src={`/api/images/${encodeURIComponent(paper.image.replace(/^\/images\//, ''))}`}
                  alt={`Thumbnail for ${paper.title}`}
                  className="w-full max-h-48 object-contain rounded-md"
                  loading="lazy"
                />
              </div>
            )}

            <SheetHeader className="px-5 pt-5 pb-3">
              <div className="flex items-start justify-between gap-2 pr-6">
                <SheetTitle className="text-sm font-semibold leading-snug flex-1">
                  {paper.title}
                </SheetTitle>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 flex-none mt-0.5"
                  onClick={() => setEditMode(e => !e)}
                  aria-label={editMode ? 'Cancel editing' : 'Edit paper'}
                  title={editMode ? 'Cancel editing' : 'Edit paper'}
                >
                  {editMode ? <X className="h-3.5 w-3.5" /> : <Pencil className="h-3.5 w-3.5" />}
                </Button>
              </div>
              {paper.authors && paper.authors.trim().length > 0 && (
                <SheetDescription className="text-xs">
                  {paper.authors}
                </SheetDescription>
              )}
            </SheetHeader>

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
                    <span className={confidenceBadge(paper.confidence_band as ConfidenceBand)}>
                      {paper.confidence_band}
                    </span>
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
                  {paper.abstract && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Abstract</p>
                      <p className="text-xs leading-relaxed text-foreground line-clamp-[12]">{paper.abstract}</p>
                    </div>
                  )}

                  {/* Links */}
                  {(paper.url || paper.pdf_url || paper.project_url) && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Links</p>
                      <div className="flex flex-wrap gap-3">
                        {paper.url && <ExternalLinkButton href={paper.url} label="Paper" />}
                        {paper.pdf_url && <ExternalLinkButton href={paper.pdf_url} label="PDF" />}
                        {paper.project_url && <ExternalLinkButton href={paper.project_url} label="Project" />}
                      </div>
                    </div>
                  )}

                  {/* Bucket actions */}
                  <BucketActions
                    paper={paper}
                    disabled={hasActiveRun}
                    onMutate={handleMutate}
                  />

                  {/* Image management */}
                  <ImageActions
                    paper={paper}
                    disabled={hasActiveRun}
                    onMutate={handleMutate}
                  />

                  {/* ID */}
                  <div>
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">ID</p>
                    <code className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                      {paper.id}
                    </code>
                  </div>

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
