import { useState, useMemo, useCallback, useRef } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  type ColumnDef,
} from '@tanstack/react-table'
import { Search, AlertCircle, FileText, ChevronDown, ChevronUp, Trash2, Lock, AlertTriangle, TriangleAlert, Plus } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
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
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { usePapers, useActiveRun } from '@/api/hooks'
import { useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { useDebounce } from '@/hooks/useDebounce'
import { bucketBadge, confidenceBadge, categoryBadge, categoryLabel } from '@/lib/tokens'
import { formatAuthors, truncate } from '@/lib/utils'
import { PaperSheet } from '@/components/papers/PaperSheet'
import { batchReprocess, setPaperBucket, addPaper, addPaperPdf } from '@/api/client'
import { toast } from 'sonner'
import type { PaperRow, Bucket, SortOption, ConfidenceBand, Category } from '@/api/types'

const BUCKETS: { label: string; value: '' | Bucket }[] = [
  { label: 'All', value: '' },
  { label: 'Verified', value: 'verified' },
  { label: 'Pending', value: 'pending' },
  { label: 'Discarded', value: 'discarded' },
]

const SORT_OPTIONS: { label: string; value: SortOption }[] = [
  { label: 'Year ↓', value: 'year_desc' },
  { label: 'Year ↑', value: 'year_asc' },
  { label: 'Title', value: 'title' },
]

const REPROCESS_FIELDS = [
  { key: 'summary',      label: 'Summary' },
  { key: 'classify',     label: 'Classify' },
  { key: 'thumbnail',    label: 'Thumbnail' },
  { key: 'affiliations', label: 'Affiliations' },
]

function TableSkeleton() {
  return (
    <div className="space-y-1.5">
      {Array.from({ length: 12 }).map((_, i) => (
        <Skeleton key={i} className="h-9 w-full" />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// SelectionToolbar — shown when ≥1 row is selected
// ---------------------------------------------------------------------------

interface SelectionToolbarProps {
  selectedIds: string[]
  hasActiveRun: boolean
  onClearSelection: () => void
  onInvalidatePapers: () => void
}

function SelectionToolbar({
  selectedIds,
  hasActiveRun,
  onClearSelection,
  onInvalidatePapers,
}: SelectionToolbarProps) {
  const n = selectedIds.length
  const [reprocessOpen, setReprocessOpen] = useState(false)
  const [reprocessConfirmOpen, setReprocessConfirmOpen] = useState(false)
  const [selectedFields, setSelectedFields] = useState<Set<string>>(new Set(['summary']))
  const [busyBucket, setBusyBucket] = useState(false)
  const [busyReprocess, setBusyReprocess] = useState(false)
  const [discardConfirmOpen, setDiscardConfirmOpen] = useState(false)

  function toggleField(key: string) {
    setSelectedFields(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key); else next.add(key)
      return next
    })
  }

  async function handlePromote() {
    if (hasActiveRun) { toast.error('A run is in progress — try again when it finishes'); return }
    setBusyBucket(true)
    let ok = 0
    for (const id of selectedIds) {
      try { await setPaperBucket(id, { bucket: 'verified' }); ok++ }
      catch (err) {
        if ((err as { status?: number }).status === 409) {
          toast.error('A run is in progress — try again when it finishes'); break
        }
      }
    }
    if (ok > 0) {
      toast.success(`Promoted ${ok} paper${ok !== 1 ? 's' : ''}`)
      onInvalidatePapers()
      onClearSelection()
    }
    setBusyBucket(false)
  }

  async function handleDiscard() {
    if (hasActiveRun) { toast.error('A run is in progress — try again when it finishes'); return }
    setBusyBucket(true)
    let ok = 0
    for (const id of selectedIds) {
      try { await setPaperBucket(id, { bucket: 'discarded', reason: 'manual_discard' }); ok++ }
      catch (err) {
        if ((err as { status?: number }).status === 409) {
          toast.error('A run is in progress — try again when it finishes'); break
        }
      }
    }
    if (ok > 0) {
      toast.success(`Discarded ${ok} paper${ok !== 1 ? 's' : ''}`)
      onInvalidatePapers()
      onClearSelection()
    }
    setBusyBucket(false)
    setDiscardConfirmOpen(false)
  }

  async function handleReprocess() {
    const fields = Array.from(selectedFields)
    if (fields.length === 0) { toast.error('Select at least one field to reprocess'); return }
    if (hasActiveRun) { toast.error('A run is in progress — try again when it finishes'); return }
    setBusyReprocess(true)
    setReprocessConfirmOpen(false)
    try {
      const res = await batchReprocess(selectedIds, fields)
      toast.info(`Reprocess job started (run ${res.run_id.slice(0, 8)}…)`)
      onClearSelection()
    } catch (err) {
      const msg = (err as { status?: number }).status === 409
        ? 'A run is in progress — try again when it finishes'
        : (err as Error).message
      toast.error(msg)
    } finally {
      setBusyReprocess(false)
    }
  }

  return (
    <div className="sticky bottom-0 z-20 bg-background border-t flex flex-wrap items-center gap-2 px-3 py-2 shadow-[0_-1px_4px_rgba(0,0,0,0.06)]">
      <span className="text-xs font-medium text-muted-foreground tabular-nums">
        {n} selected
      </span>
      <Button
        size="sm"
        variant="ghost"
        className="h-7 text-xs"
        onClick={onClearSelection}
      >
        Clear
      </Button>
      <div className="flex-1" />

      {/* Promote */}
      <Button
        size="sm"
        variant="outline"
        disabled={busyBucket || hasActiveRun}
        onClick={handlePromote}
        className="h-7 text-xs gap-1"
      >
        <ChevronUp className="h-3 w-3" />
        Promote {n}
      </Button>

      {/* Discard */}
      <Button
        size="sm"
        variant="outline"
        disabled={busyBucket || hasActiveRun}
        onClick={() => setDiscardConfirmOpen(true)}
        className="h-7 text-xs gap-1 text-destructive hover:text-destructive border-destructive/30"
      >
        <Trash2 className="h-3 w-3" />
        Discard {n}
      </Button>

      {/* Reprocess dropdown */}
      <div className="relative">
        <Button
          size="sm"
          variant="outline"
          disabled={busyReprocess || hasActiveRun}
          onClick={() => setReprocessOpen(o => !o)}
          className="h-7 text-xs gap-1"
        >
          Reprocess
          <ChevronDown className="h-3 w-3" />
        </Button>
        {reprocessOpen && (
          <div className="absolute right-0 bottom-full mb-1 z-30 w-52 rounded-md border bg-popover shadow-md p-2 space-y-1">
            <p className="text-xs font-semibold text-muted-foreground px-1 pb-1">Fields to reprocess</p>
            {REPROCESS_FIELDS.map(f => (
              <label key={f.key} className="flex items-center gap-2 px-1 py-0.5 text-xs cursor-pointer hover:bg-accent rounded-sm">
                <input
                  type="checkbox"
                  checked={selectedFields.has(f.key)}
                  onChange={() => toggleField(f.key)}
                  className="h-3 w-3"
                />
                {f.label}
              </label>
            ))}
            <div className="pt-1 border-t">
              <Button
                size="sm"
                className="w-full h-7 text-xs"
                disabled={selectedFields.size === 0}
                onClick={() => { setReprocessOpen(false); setReprocessConfirmOpen(true) }}
              >
                Reprocess {n} papers
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Discard confirm */}
      <AlertDialog open={discardConfirmOpen} onOpenChange={setDiscardConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Discard {n} paper{n !== 1 ? 's' : ''}?</AlertDialogTitle>
            <AlertDialogDescription>
              This will move all selected papers to the discarded bucket.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDiscardConfirmOpen(false)}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDiscard}>Discard</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Reprocess confirm */}
      <AlertDialog open={reprocessConfirmOpen} onOpenChange={setReprocessConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Reprocess {n} paper{n !== 1 ? 's' : ''}?</AlertDialogTitle>
            <AlertDialogDescription>
              This will overwrite curated fields ({Array.from(selectedFields).join(', ')}) and
              may spend LLM credits. A pipeline run will be started; you cannot start another run
              until it completes.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setReprocessConfirmOpen(false)}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleReprocess}>
              Reprocess
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Papers page
// ---------------------------------------------------------------------------

export function Papers() {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const [bucket, setBucket] = useState<'' | Bucket>('')
  const [searchInput, setSearchInput] = useState('')
  const [sort, setSort] = useState<SortOption>('year_desc')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedRowIds, setSelectedRowIds] = useState<Set<string>>(new Set())
  const [needsAttention, setNeedsAttention] = useState(false)

  // Add paper dialog state
  const [addOpen, setAddOpen] = useState(false)
  const [linkUrl, setLinkUrl] = useState('')
  const [pdfTitle, setPdfTitle] = useState('')
  const [pdfArxiv, setPdfArxiv] = useState('')
  const [pdfDoi, setPdfDoi] = useState('')
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { data: activeRunData } = useActiveRun()
  const hasActiveRun = !!(activeRunData?.active)

  function resetAddDialog() {
    setLinkUrl('')
    setPdfTitle('')
    setPdfArxiv('')
    setPdfDoi('')
    setPdfFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  async function submitLink() {
    if (!linkUrl.trim()) { toast.error('Enter a URL'); return }
    setSubmitting(true)
    try {
      await addPaper(linkUrl.trim())
      toast.success('Added — review the candidate at the gate')
      setAddOpen(false)
      resetAddDialog()
      navigate('/runs')
    } catch (e) {
      const status = (e as { status?: number }).status
      toast.error(status === 409 ? 'A run is already active — wait for it to finish.' : ((e as Error).message || 'Add failed'))
    } finally {
      setSubmitting(false)
    }
  }

  async function submitPdf() {
    if (!pdfTitle.trim()) { toast.error('Title is required'); return }
    if (!pdfFile) { toast.error('Select a PDF file'); return }
    setSubmitting(true)
    try {
      await addPaperPdf({
        title: pdfTitle.trim(),
        arxiv_id: pdfArxiv.trim() || undefined,
        doi: pdfDoi.trim() || undefined,
        file: pdfFile,
      })
      toast.success('Added — review the candidate at the gate')
      setAddOpen(false)
      resetAddDialog()
      navigate('/runs')
    } catch (e) {
      const status = (e as { status?: number }).status
      toast.error(status === 409 ? 'A run is already active — wait for it to finish.' : ((e as Error).message || 'Add failed'))
    } finally {
      setSubmitting(false)
    }
  }

  const q = useDebounce(searchInput, 300)

  const { data: papers, isLoading, error } = usePapers({
    bucket: bucket || undefined,
    q: q || undefined,
    sort,
  })

  const invalidatePapers = useCallback(() => {
    qc.invalidateQueries({ queryKey: ['papers'] })
  }, [qc])

  const clearSelection = useCallback(() => setSelectedRowIds(new Set()), [])

  // Client-side "needs attention" filter on top of server-side results
  const filteredPapers = useMemo(() => {
    if (!needsAttention) return papers ?? []
    return (papers ?? []).filter(p => (p.missing ?? []).length > 0)
  }, [papers, needsAttention])

  const needsAttentionCount = useMemo(
    () => (papers ?? []).filter(p => (p.missing ?? []).length > 0).length,
    [papers],
  )

  const allIds = useMemo(() => filteredPapers.map(p => p.id), [filteredPapers])
  const allSelected = allIds.length > 0 && allIds.every(id => selectedRowIds.has(id))
  const someSelected = !allSelected && allIds.some(id => selectedRowIds.has(id))

  function toggleSelectAll() {
    if (allSelected) {
      setSelectedRowIds(new Set())
    } else {
      setSelectedRowIds(new Set(allIds))
    }
  }

  function toggleRow(id: string, e: React.MouseEvent | React.ChangeEvent) {
    e.stopPropagation()
    setSelectedRowIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }

  const columns = useMemo<ColumnDef<PaperRow>[]>(
    () => [
      {
        id: 'select',
        header: () => (
          <input
            type="checkbox"
            checked={allSelected}
            ref={el => { if (el) el.indeterminate = someSelected }}
            onChange={toggleSelectAll}
            className="h-3.5 w-3.5 cursor-pointer"
            aria-label="Select all papers"
            onClick={e => e.stopPropagation()}
          />
        ),
        cell: ({ row }) => (
          <input
            type="checkbox"
            checked={selectedRowIds.has(row.original.id)}
            onChange={e => toggleRow(row.original.id, e)}
            onClick={e => e.stopPropagation()}
            className="h-3.5 w-3.5 cursor-pointer"
            aria-label={`Select ${row.original.title}`}
          />
        ),
        size: 36,
      },
      {
        id: 'title',
        header: 'Title',
        accessorKey: 'title',
        size: 260,
        cell: ({ getValue }) => {
          const title = getValue<string>()
          const truncated = truncate(title, 80)
          if (title === truncated) return <span className="font-medium text-sm">{title}</span>
          return (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="font-medium text-sm cursor-default">{truncated}</span>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-sm">{title}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )
        },
      },
      {
        id: 'flags',
        header: '',
        size: 44,
        cell: ({ row }) => {
          const { manual_override } = row.original
          // Defensive: an older backend may omit `missing` (FE/BE version skew) — never crash the table.
          const missing = row.original.missing ?? []
          if (!manual_override && missing.length === 0) return null
          return (
            <TooltipProvider>
              <div className="flex items-center gap-1">
                {manual_override && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Lock className="h-3 w-3 text-muted-foreground flex-none" aria-label="Curator-locked" />
                    </TooltipTrigger>
                    <TooltipContent side="top">Curator-locked — pipeline won&apos;t overwrite</TooltipContent>
                  </Tooltip>
                )}
                {missing.length > 0 && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <AlertTriangle className="h-3 w-3 text-amber-500 flex-none" aria-label="Missing metadata" />
                    </TooltipTrigger>
                    <TooltipContent side="top">Missing: {missing.join(', ')}</TooltipContent>
                  </Tooltip>
                )}
              </div>
            </TooltipProvider>
          )
        },
      },
      {
        id: 'authors',
        header: 'Authors',
        accessorKey: 'authors',
        size: 180,
        cell: ({ getValue }) => {
          const authors = getValue<string>() ?? ''
          const names = authors ? authors.split(/,\s*/).filter(Boolean) : []
          const formatted = formatAuthors(authors)
          if (names.length <= 3) {
            return <span className="text-xs text-muted-foreground">{formatted}</span>
          }
          return (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-xs text-muted-foreground cursor-default">{formatted}</span>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-xs">{authors}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )
        },
      },
      {
        id: 'venue',
        header: 'Venue',
        accessorKey: 'venue',
        size: 120,
        cell: ({ getValue }) => (
          <span className="text-xs text-muted-foreground">
            {getValue<string | null>() ?? '—'}
          </span>
        ),
      },
      {
        id: 'year',
        header: 'Year',
        accessorKey: 'year',
        size: 56,
        cell: ({ getValue }) => (
          <span className="text-xs tabular-nums">
            {getValue<number | null>() ?? '—'}
          </span>
        ),
      },
      {
        id: 'category',
        header: 'Category',
        accessorKey: 'category',
        size: 110,
        cell: ({ getValue }) => {
          const cat = getValue<Category>()
          return <span className={categoryBadge(cat)}>{categoryLabel(cat)}</span>
        },
      },
      {
        id: 'bucket',
        header: 'Bucket',
        accessorKey: 'bucket',
        size: 90,
        cell: ({ getValue }) => {
          const b = getValue<Bucket>()
          return <span className={bucketBadge(b)}>{b}</span>
        },
      },
      {
        id: 'confidence',
        header: 'Conf.',
        accessorKey: 'confidence_band',
        size: 72,
        cell: ({ getValue }) => {
          const band = getValue<ConfidenceBand>()
          return <span className={confidenceBadge(band)}>{band}</span>
        },
      },
      {
        id: 'source',
        header: 'Source',
        accessorKey: 'source',
        size: 80,
        cell: ({ getValue }) => (
          <span className="text-xs font-mono text-muted-foreground truncate">
            {getValue<string | null>() ?? '—'}
          </span>
        ),
      },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [allSelected, someSelected, selectedRowIds]
  )

  const table = useReactTable({
    data: filteredPapers,
    columns,
    getCoreRowModel: getCoreRowModel(),
  })

  // Ordered IDs matching the visible table rows (respects needs-attention filter + sort)
  const orderedIds = useMemo(
    () => table.getRowModel().rows.map(r => r.original.id),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [table.getRowModel().rows],
  )

  const currentIndex = selectedId !== null ? orderedIds.indexOf(selectedId) : -1
  const hasPrev = currentIndex > 0
  const hasNext = currentIndex !== -1 && currentIndex < orderedIds.length - 1

  const handlePrev = useCallback(() => {
    if (currentIndex > 0) setSelectedId(orderedIds[currentIndex - 1])
  }, [currentIndex, orderedIds])

  const handleNext = useCallback(() => {
    if (currentIndex !== -1 && currentIndex < orderedIds.length - 1)
      setSelectedId(orderedIds[currentIndex + 1])
  }, [currentIndex, orderedIds])

  const selectedIdsList = useMemo(() => Array.from(selectedRowIds), [selectedRowIds])
  const hasSelection = selectedIdsList.length > 0

  return (
    <div className="flex flex-col gap-3 h-full max-h-[calc(100vh-96px)]">
      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Bucket filter as pill tabs */}
        <div
          className="inline-flex h-8 items-center rounded-lg bg-muted p-1 gap-0.5"
          role="tablist"
          aria-label="Filter by bucket"
        >
          {BUCKETS.map(({ label, value }) => (
            <button
              key={value}
              role="tab"
              aria-selected={bucket === value}
              onClick={() => setBucket(value)}
              className={[
                'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1 text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                bucket === value
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground',
              ].join(' ')}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative flex-1 min-w-[160px] max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" aria-hidden="true" />
          <Input
            type="search"
            placeholder="Search papers…"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="pl-8 h-8 text-xs"
            aria-label="Search papers"
          />
        </div>

        {/* Sort */}
        <Select value={sort} onValueChange={(v) => setSort(v as SortOption)}>
          <SelectTrigger className="w-28 h-8 text-xs" aria-label="Sort papers">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SORT_OPTIONS.map(({ label, value }) => (
              <SelectItem key={value} value={value}>{label}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Needs attention toggle */}
        <button
          type="button"
          onClick={() => setNeedsAttention(v => !v)}
          aria-pressed={needsAttention}
          className={[
            'inline-flex items-center gap-1.5 h-8 rounded-md px-3 text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring border',
            needsAttention
              ? 'bg-amber-50 text-amber-700 border-amber-300 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-700'
              : 'bg-background text-muted-foreground border-border hover:text-foreground hover:border-border',
          ].join(' ')}
          aria-label={`Filter by needs attention${needsAttentionCount > 0 ? ` (${needsAttentionCount})` : ''}`}
        >
          <TriangleAlert className="h-3.5 w-3.5" aria-hidden="true" />
          Needs attention
          {needsAttentionCount > 0 && (
            <span className="tabular-nums">{needsAttentionCount}</span>
          )}
        </button>

        {/* Count */}
        {!isLoading && papers && (
          <span className="text-xs text-muted-foreground tabular-nums ml-auto">
            {needsAttention ? `${filteredPapers.length} / ${papers.length}` : papers.length} papers
          </span>
        )}

        {/* Add paper button */}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs gap-1"
                  disabled={hasActiveRun}
                  onClick={() => setAddOpen(true)}
                  aria-label="Add paper"
                >
                  <Plus className="h-3.5 w-3.5" aria-hidden="true" />
                  Add paper
                </Button>
              </span>
            </TooltipTrigger>
            {hasActiveRun && (
              <TooltipContent side="bottom">A run is already active</TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Add paper dialog */}
      <Dialog open={addOpen} onOpenChange={(open) => { setAddOpen(open); if (!open) resetAddDialog() }}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Add paper</DialogTitle>
            <DialogDescription>
              Add a paper by URL or upload a PDF. This starts a gated manual-add run.
            </DialogDescription>
          </DialogHeader>
          <div className="px-6 pb-6">
            <Tabs defaultValue="link">
              <TabsList className="mb-4">
                <TabsTrigger value="link">By link</TabsTrigger>
                <TabsTrigger value="pdf">By PDF</TabsTrigger>
              </TabsList>

              {/* By link tab */}
              <TabsContent value="link">
                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-medium text-foreground mb-1.5 block">URL</label>
                    <Input
                      type="url"
                      placeholder="https://arxiv.org/abs/…"
                      value={linkUrl}
                      onChange={e => setLinkUrl(e.target.value)}
                      className="h-8 text-xs"
                      onKeyDown={e => { if (e.key === 'Enter') submitLink() }}
                    />
                  </div>
                  <Button
                    size="sm"
                    className="w-full h-8 text-xs"
                    disabled={submitting || hasActiveRun || !linkUrl.trim()}
                    onClick={submitLink}
                  >
                    {submitting ? 'Adding…' : 'Add paper'}
                  </Button>
                </div>
              </TabsContent>

              {/* By PDF tab */}
              <TabsContent value="pdf">
                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-medium text-foreground mb-1.5 block">
                      Title <span className="text-destructive">*</span>
                    </label>
                    <Input
                      placeholder="Paper title"
                      value={pdfTitle}
                      onChange={e => setPdfTitle(e.target.value)}
                      className="h-8 text-xs"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-foreground mb-1.5 block">arXiv ID <span className="text-muted-foreground font-normal">(optional)</span></label>
                    <Input
                      placeholder="e.g. 2301.00001"
                      value={pdfArxiv}
                      onChange={e => setPdfArxiv(e.target.value)}
                      className="h-8 text-xs"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-foreground mb-1.5 block">DOI <span className="text-muted-foreground font-normal">(optional)</span></label>
                    <Input
                      placeholder="e.g. 10.1234/example"
                      value={pdfDoi}
                      onChange={e => setPdfDoi(e.target.value)}
                      className="h-8 text-xs"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-foreground mb-1.5 block">
                      PDF file <span className="text-destructive">*</span>
                    </label>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="application/pdf"
                      className="text-xs w-full cursor-pointer file:mr-2 file:h-7 file:cursor-pointer file:rounded-md file:border file:border-input file:bg-background file:px-2 file:text-xs file:font-medium file:text-foreground hover:file:bg-accent"
                      onChange={e => setPdfFile(e.target.files?.[0] ?? null)}
                    />
                  </div>
                  <Button
                    size="sm"
                    className="w-full h-8 text-xs"
                    disabled={submitting || hasActiveRun || !pdfTitle.trim() || !pdfFile}
                    onClick={submitPdf}
                  >
                    {submitting ? 'Adding…' : 'Add paper'}
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </DialogContent>
      </Dialog>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-destructive text-sm">
          <AlertCircle className="h-4 w-4 flex-none" aria-hidden="true" />
          <span>Failed to load papers: {(error as Error).message}</span>
        </div>
      )}

      {/* Table + toolbar wrapper */}
      <div className="flex-1 flex flex-col overflow-hidden rounded-md border bg-card">
        <div className="flex-1 overflow-auto">
          {isLoading ? (
            <div className="p-3">
              <TableSkeleton />
            </div>
          ) : (
            <table
              className="sticky-header w-full text-left border-collapse"
              aria-label="Papers table"
            >
              <thead>
                {table.getHeaderGroups().map(hg => (
                  <tr key={hg.id}>
                    {hg.headers.map(h => (
                      <th
                        key={h.id}
                        style={{ width: h.getSize() }}
                        className="px-3 py-2 text-xs font-medium text-muted-foreground whitespace-nowrap select-none"
                      >
                        {flexRender(h.column.columnDef.header, h.getContext())}
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.length === 0 ? (
                  <tr>
                    <td
                      colSpan={columns.length}
                      className="text-center py-16 text-sm text-muted-foreground"
                    >
                      <FileText className="h-8 w-8 mx-auto mb-2 opacity-30" aria-hidden="true" />
                      No papers match
                    </td>
                  </tr>
                ) : (
                  table.getRowModel().rows.map(row => (
                    <tr
                      key={row.id}
                      onClick={() => setSelectedId(row.original.id)}
                      className={[
                        'border-t border-border/50 hover:bg-muted/50 cursor-pointer transition-colors',
                        selectedRowIds.has(row.original.id) ? 'bg-muted/30' : '',
                      ].join(' ')}
                      tabIndex={0}
                      role="button"
                      aria-label={`View details for ${row.original.title}`}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          setSelectedId(row.original.id)
                        }
                      }}
                    >
                      {row.getVisibleCells().map(cell => (
                        <td
                          key={cell.id}
                          style={{ width: cell.column.getSize() }}
                          className="px-3 py-2 max-w-0 overflow-hidden"
                        >
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
        </div>

        {/* Sticky selection toolbar */}
        {hasSelection && (
          <SelectionToolbar
            selectedIds={selectedIdsList}
            hasActiveRun={hasActiveRun}
            onClearSelection={clearSelection}
            onInvalidatePapers={invalidatePapers}
          />
        )}
      </div>

      {/* Paper detail drawer */}
      <PaperSheet
        paperId={selectedId}
        onClose={() => setSelectedId(null)}
        onPrev={handlePrev}
        onNext={handleNext}
        hasPrev={hasPrev}
        hasNext={hasNext}
      />
    </div>
  )
}
