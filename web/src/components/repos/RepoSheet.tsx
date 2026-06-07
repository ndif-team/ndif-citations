import { ExternalLink, AlertCircle, Pencil, X, Trash2, GitFork, Lock } from 'lucide-react'
import { useState, useCallback, useRef } from 'react'
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
import { useRepo, useActiveRun } from '@/api/hooks'
import { useQueryClient } from '@tanstack/react-query'
import { categoryBadge, repoTypeBadge, repoTypeLabel } from '@/lib/tokens'
import { editRepo, excludeRepo } from '@/api/client'
import type { RepoDetail, RepoType } from '@/api/types'
import { toast } from 'sonner'

interface Props {
  owner: string | null
  repo: string | null
  onClose: () => void
}

const REPO_TYPE_OPTIONS: { value: RepoType; label: string }[] = [
  { value: 'research',   label: 'Research' },
  { value: 'course',     label: 'Course' },
  { value: 'experiment', label: 'Experiment' },
]

// ---------------------------------------------------------------------------
// Small helper
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

// ---------------------------------------------------------------------------
// EditForm
// ---------------------------------------------------------------------------

interface EditFormProps {
  detail: RepoDetail
  disabled: boolean
  saving: boolean
  onSave: (payload: { repo_type?: RepoType; linked_paper_url?: string | null; description?: string | null }) => void
  onCancel: () => void
}

function EditForm({ detail, disabled, saving, onSave, onCancel }: EditFormProps) {
  const [repoType, setRepoType] = useState<RepoType>(detail.repo_type)
  const [linkedPaperUrl, setLinkedPaperUrl] = useState<string>(detail.linked_paper_url ?? '')
  const [description, setDescription] = useState<string>(detail.description ?? '')

  function handleSave() {
    const payload: { repo_type?: RepoType; linked_paper_url?: string | null; description?: string | null } = {}

    if (repoType !== detail.repo_type) {
      payload.repo_type = repoType
    }

    // Treat changed URL: empty string → null (clear); otherwise pass string
    const newLinkedUrl = linkedPaperUrl.trim() === '' ? null : linkedPaperUrl.trim()
    const origLinkedUrl = detail.linked_paper_url ?? null
    if (newLinkedUrl !== origLinkedUrl) {
      payload.linked_paper_url = newLinkedUrl
    }

    // Treat changed description: empty string → null (clear); otherwise pass string
    const newDesc = description.trim() === '' ? null : description.trim()
    const origDesc = detail.description ?? null
    if (newDesc !== origDesc) {
      payload.description = newDesc
    }

    if (Object.keys(payload).length === 0) {
      onCancel()
      return
    }

    onSave(payload)
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-3">
        {/* repo_type — all non-empty real values, no empty option */}
        <div className="space-y-0.5">
          <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Type
          </label>
          <Select
            value={repoType}
            onValueChange={(v) => setRepoType(v as RepoType)}
            disabled={disabled || saving}
          >
            <SelectTrigger className="h-7 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {REPO_TYPE_OPTIONS.map(opt => (
                <SelectItem key={opt.value} value={opt.value} className="text-xs">
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* linked_paper_url */}
        <div className="space-y-0.5">
          <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Linked paper URL
          </label>
          <Input
            type="text"
            value={linkedPaperUrl}
            onChange={e => setLinkedPaperUrl(e.target.value)}
            disabled={disabled || saving}
            placeholder="https://arxiv.org/abs/… (empty to clear)"
            className="h-7 text-xs"
          />
        </div>

        {/* description */}
        <div className="space-y-0.5">
          <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Description
          </label>
          <textarea
            value={description}
            onChange={e => setDescription(e.target.value)}
            disabled={disabled || saving}
            placeholder="Short description (empty to clear)"
            rows={3}
            className="flex w-full rounded-md border border-input bg-background px-3 py-1.5 text-xs shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 resize-y"
          />
        </div>
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

// ---------------------------------------------------------------------------
// Main RepoSheet
// ---------------------------------------------------------------------------

export function RepoSheet({ owner, repo, onClose }: Props) {
  const qc = useQueryClient()
  const { data: detail, isLoading, error } = useRepo(owner, repo)
  const { data: activeRunData } = useActiveRun()
  const hasActiveRun = !!(activeRunData?.active)

  const [editMode, setEditMode] = useState(false)
  const [saving, setSaving] = useState(false)
  const [excludeOpen, setExcludeOpen] = useState(false)

  const isOpen = !!(owner && repo)

  // Reset edit mode when the selected repo changes
  const prevKey = useRef<string | null>(null)
  const key = owner && repo ? `${owner}/${repo}` : null
  if (key !== prevKey.current) {
    prevKey.current = key
    // state reset deferred — handled by key change
  }

  const handleMutate = useCallback(
    (updater: (prev: RepoDetail) => RepoDetail) => {
      if (!owner || !repo) return
      qc.setQueryData<RepoDetail>(['repo', owner, repo], old => {
        if (!old) return old
        return updater(old)
      })
      qc.invalidateQueries({ queryKey: ['repos'] })
    },
    [owner, repo, qc],
  )

  const handleSave = useCallback(
    async (payload: { repo_type?: RepoType; linked_paper_url?: string | null; description?: string | null }) => {
      if (!owner || !repo) return
      setSaving(true)
      try {
        const updated = await editRepo(owner, repo, payload)
        handleMutate(() => updated)
        setEditMode(false)
        toast.success('Saved')
      } catch (err) {
        const status = (err as { status?: number }).status
        if (status === 409) {
          toast.error('A run is in progress — try again when it finishes')
        } else if (status === 422) {
          toast.error(`Validation error: ${(err as Error).message}`)
        } else {
          toast.error((err as Error).message)
        }
      } finally {
        setSaving(false)
      }
    },
    [owner, repo, handleMutate],
  )

  const handleExclude = useCallback(async () => {
    if (!owner || !repo) return
    setExcludeOpen(false)
    try {
      await excludeRepo(owner, repo)
      qc.invalidateQueries({ queryKey: ['repos'] })
      toast.success(`Excluded ${owner}/${repo}`)
      onClose()
    } catch (err) {
      const status = (err as { status?: number }).status
      if (status === 409) {
        toast.error('A run is in progress — try again when it finishes')
      } else {
        toast.error((err as Error).message)
      }
    }
  }, [owner, repo, qc, onClose])

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
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-4/6" />
          </div>
        )}

        {error && (
          <div className="p-6 flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="h-4 w-4 flex-none" aria-hidden="true" />
            <span>Failed to load repo: {(error as Error).message}</span>
          </div>
        )}

        {detail && (
          <>
            <SheetHeader className="px-5 pt-5 pb-3">
              <div className="flex items-start justify-between gap-2 pr-6">
                <div className="flex-1 min-w-0">
                  <SheetTitle className="text-sm font-semibold leading-snug font-mono flex items-center gap-1.5 flex-wrap">
                    {detail.is_fork && (
                      <GitFork className="h-3.5 w-3.5 text-muted-foreground flex-none" aria-label="Fork" />
                    )}
                    {detail.archived && (
                      <Lock className="h-3.5 w-3.5 text-muted-foreground flex-none" aria-label="Archived" />
                    )}
                    <a
                      href={detail.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline text-primary"
                    >
                      {detail.owner}/{detail.repo}
                    </a>
                  </SheetTitle>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 flex-none mt-0.5"
                  onClick={() => setEditMode(e => !e)}
                  disabled={hasActiveRun}
                  aria-label={editMode ? 'Cancel editing' : 'Edit repo'}
                  title={editMode ? 'Cancel editing' : 'Edit repo'}
                >
                  {editMode ? <X className="h-3.5 w-3.5" /> : <Pencil className="h-3.5 w-3.5" />}
                </Button>
              </div>
              {detail.description && (
                <SheetDescription className="text-xs">
                  {detail.description}
                </SheetDescription>
              )}
            </SheetHeader>

            <div className="px-5 pb-5 space-y-4 flex-1">
              {editMode ? (
                <EditForm
                  detail={detail}
                  disabled={hasActiveRun}
                  saving={saving}
                  onSave={handleSave}
                  onCancel={() => setEditMode(false)}
                />
              ) : (
                <>
                  {/* Type + Category badges */}
                  <div className="flex flex-wrap gap-1.5">
                    <span className={repoTypeBadge(detail.repo_type)}>
                      {repoTypeLabel(detail.repo_type)}
                    </span>
                    <span className={categoryBadge(detail.category as Parameters<typeof categoryBadge>[0])}>
                      {detail.category}
                    </span>
                    {detail.manual_override && (
                      <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset bg-orange-50 text-orange-800 ring-orange-200 dark:bg-orange-950 dark:text-orange-300 dark:ring-orange-800">
                        manual override
                      </span>
                    )}
                    {detail.archived && (
                      <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset bg-slate-100 text-slate-600 ring-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:ring-slate-700">
                        archived
                      </span>
                    )}
                  </div>

                  {/* Stats row */}
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                    {detail.stars !== null && (
                      <span>⭐ <span className="tabular-nums font-medium text-foreground">{detail.stars.toLocaleString()}</span> stars</span>
                    )}
                    {detail.forks !== null && (
                      <span>⑂ <span className="tabular-nums font-medium text-foreground">{detail.forks.toLocaleString()}</span> forks</span>
                    )}
                    {detail.language && (
                      <span><span className="font-medium text-foreground">{detail.language}</span></span>
                    )}
                    {detail.last_commit && (
                      <span>Last commit: <span className="font-medium text-foreground">{detail.last_commit}</span></span>
                    )}
                  </div>

                  {/* Linked paper */}
                  {detail.linked_paper_url && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Linked paper</p>
                      <ExternalLinkButton href={detail.linked_paper_url} label={detail.linked_paper_url} />
                      {detail.linked_paper_tier !== null && (
                        <span className="ml-2 text-xs text-muted-foreground">(tier {detail.linked_paper_tier})</span>
                      )}
                    </div>
                  )}

                  {/* Fork info */}
                  {detail.is_fork && detail.parent_full_name && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Forked from</p>
                      <a
                        href={`https://github.com/${detail.parent_full_name}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs font-mono text-primary hover:underline"
                      >
                        {detail.parent_full_name}
                      </a>
                    </div>
                  )}

                  {/* Topics */}
                  {detail.topics.length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Topics</p>
                      <div className="flex flex-wrap gap-1">
                        {detail.topics.map(t => (
                          <span
                            key={t}
                            className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium bg-muted text-muted-foreground ring-1 ring-inset ring-border"
                          >
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Classification reason */}
                  {detail.classification_reason && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Classification reason</p>
                      <p className="text-xs text-muted-foreground">{detail.classification_reason}</p>
                    </div>
                  )}

                  {/* README arXiv IDs */}
                  {detail.readme_arxiv_ids.length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">README arXiv IDs</p>
                      <div className="flex flex-col gap-1">
                        {detail.readme_arxiv_ids.map(id => (
                          <a
                            key={id}
                            href={`https://arxiv.org/abs/${id}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs font-mono text-primary hover:underline"
                          >
                            {id}
                          </a>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Exclude */}
                  <div className="pt-2 border-t">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                      Danger zone
                    </p>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={hasActiveRun}
                      onClick={() => setExcludeOpen(true)}
                      className="h-7 text-xs gap-1 text-destructive hover:text-destructive border-destructive/30 hover:border-destructive/60"
                    >
                      <Trash2 className="h-3 w-3" />
                      Exclude repo
                    </Button>
                  </div>

                  <AlertDialog open={excludeOpen} onOpenChange={setExcludeOpen}>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Exclude this repo?</AlertDialogTitle>
                        <AlertDialogDescription>
                          <strong>{detail.owner}/{detail.repo}</strong> will be removed and added to the excluded list so future runs skip it.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel onClick={() => setExcludeOpen(false)}>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={handleExclude}>Exclude</AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
