import { useState, useEffect } from 'react'
import { AlertCircle, Upload, UploadCloud, Download } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
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
import { usePublishTarget, useActiveRun } from '@/api/hooks'
import { useQueryClient } from '@tanstack/react-query'
import { putPublishTarget, runPublish, exportXlsx } from '@/api/client'
import { toast } from 'sonner'
import { SectionLabel, toastApiError, extractApiError } from '@/lib/apiError'
import type { PublishDryRunResponse, PublishResponse } from '@/api/types'

function isDryRunResponse(r: PublishDryRunResponse | PublishResponse): r is PublishDryRunResponse {
  return (r as PublishDryRunResponse).papers !== undefined
}

type DiffRecord = Record<string, unknown> & { _changed_fields?: string[] }

/** Identity label for a diff record: papers have a title, repos owner/repo. */
function diffLabel(item: DiffRecord): string {
  if (typeof item.title === 'string') return item.title
  if (typeof item.owner === 'string' && typeof item.repo === 'string') return `${item.owner}/${item.repo}`
  return JSON.stringify(item).slice(0, 60)
}

function DiffBucket({
  label, items, tone,
}: { label: string; items: DiffRecord[]; tone: 'added' | 'changed' | 'removed' }) {
  if (!items.length) return null
  const toneCls =
    tone === 'added' ? 'text-green-700 dark:text-green-400'
    : tone === 'changed' ? 'text-amber-700 dark:text-amber-400'
    : 'text-red-700 dark:text-red-400'
  const sigil = tone === 'added' ? '+' : tone === 'changed' ? '~' : '−'
  return (
    <details className="group">
      <summary className={`cursor-pointer select-none ${toneCls}`}>
        {sigil}{items.length} {label}
      </summary>
      <ul className="mt-1 ml-4 space-y-0.5">
        {items.map((item, i) => (
          <li key={i} className="text-foreground/80 flex flex-wrap items-baseline gap-1.5">
            <span className="truncate max-w-md">{diffLabel(item)}</span>
            {tone === 'changed' && Array.isArray(item._changed_fields) && item._changed_fields.length > 0 && (
              <span className="text-2xs text-muted-foreground">
                ({item._changed_fields.join(', ')})
              </span>
            )}
          </li>
        ))}
      </ul>
    </details>
  )
}

function DiffSummary({ result }: { result: PublishDryRunResponse }) {
  const { papers, repos, images } = result
  const len = (a: unknown[] | undefined) => (Array.isArray(a) ? a.length : 0)
  const total =
    len(papers?.added) + len(papers?.changed) + len(papers?.removed) +
    len(repos?.added) + len(repos?.changed) + len(repos?.removed) +
    len(images?.new) + len(images?.changed)
  return (
    <div className="rounded-md border bg-muted/30 p-3 text-xs space-y-2">
      <p className="font-medium text-foreground">
        Dry-run diff {total === 0 && <span className="font-normal text-muted-foreground">— nothing to publish, site is up to date</span>}
      </p>
      {(len(papers?.added) + len(papers?.changed) + len(papers?.removed)) > 0 && (
        <div className="space-y-1">
          <p className="text-muted-foreground font-medium">Papers</p>
          <DiffBucket label="added" tone="added" items={(papers?.added ?? []) as DiffRecord[]} />
          <DiffBucket label="changed" tone="changed" items={(papers?.changed ?? []) as DiffRecord[]} />
          <DiffBucket label="removed" tone="removed" items={(papers?.removed ?? []) as DiffRecord[]} />
        </div>
      )}
      {(len(repos?.added) + len(repos?.changed) + len(repos?.removed)) > 0 && (
        <div className="space-y-1">
          <p className="text-muted-foreground font-medium">Repos</p>
          <DiffBucket label="added" tone="added" items={(repos?.added ?? []) as DiffRecord[]} />
          <DiffBucket label="changed" tone="changed" items={(repos?.changed ?? []) as DiffRecord[]} />
          <DiffBucket label="removed" tone="removed" items={(repos?.removed ?? []) as DiffRecord[]} />
        </div>
      )}
      {(len(images?.new) + len(images?.changed)) > 0 && (
        <div className="space-y-1">
          <p className="text-muted-foreground font-medium">Images</p>
          <DiffBucket label="new" tone="added" items={(images?.new ?? []).map((f) => ({ title: f })) as DiffRecord[]} />
          <DiffBucket label="changed" tone="changed" items={(images?.changed ?? []).map((f) => ({ title: f })) as DiffRecord[]} />
        </div>
      )}
    </div>
  )
}

export function Publish() {
  const { data: activeRunData } = useActiveRun()
  const hasActiveRun = !!(activeRunData?.active)

  const { data: targetData, isLoading: targetLoading, error: targetError } = usePublishTarget()
  const qc = useQueryClient()

  const [targetInput, setTargetInput] = useState('')
  const [settingTarget, setSettingTarget] = useState(false)

  // What-to-publish scope (both default on).
  const [pubPapers, setPubPapers] = useState(true)
  const [pubRepos, setPubRepos] = useState(true)
  const noScope = !pubPapers && !pubRepos

  const [dryRunBusy, setDryRunBusy] = useState(false)
  const [dryRunResult, setDryRunResult] = useState<PublishDryRunResponse | null>(null)

  const [publishBusy, setPublishBusy] = useState(false)
  const [publishResult, setPublishResult] = useState<PublishResponse | null>(null)
  const [publishConfirmOpen, setPublishConfirmOpen] = useState(false)

  const [exporting, setExporting] = useState(false)
  async function handleExport() {
    setExporting(true)
    try {
      await exportXlsx()
      toast.success('Exported .xlsx')
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setExporting(false)
    }
  }

  // Sync target input when data loads
  useEffect(() => {
    if (targetData) {
      setTargetInput(targetData.configured ?? targetData.detected ?? '')
    }
  }, [targetData])

  async function handleSetTarget() {
    if (!targetInput.trim()) return
    setSettingTarget(true)
    try {
      const res = await putPublishTarget(targetInput.trim())
      qc.setQueryData(['publish', 'target'], (old: typeof targetData) => old
        ? { ...old, configured: res.publish_target, valid: true }
        : old
      )
      qc.invalidateQueries({ queryKey: ['publish', 'target'] })
      toast.success('Publish target set')
    } catch (err) {
      toastApiError(err)
    } finally {
      setSettingTarget(false)
    }
  }

  async function handleDryRun() {
    setDryRunBusy(true)
    setDryRunResult(null)
    try {
      const res = await runPublish({ dry_run: true, publish_papers: pubPapers, publish_repos: pubRepos })
      if (isDryRunResponse(res)) {
        setDryRunResult(res)
      } else {
        toast.info('Dry run complete')
      }
    } catch (err) {
      const { status } = extractApiError(err)
      if (status === 400) toast.error('Configure a publish target first')
      else toastApiError(err)
    } finally {
      setDryRunBusy(false)
    }
  }

  async function handlePublish() {
    setPublishConfirmOpen(false)
    setPublishBusy(true)
    setPublishResult(null)
    try {
      const res = await runPublish({ dry_run: false, publish_papers: pubPapers, publish_repos: pubRepos })
      if (!isDryRunResponse(res)) {
        setPublishResult(res)
        toast.success('Published successfully')
      }
    } catch (err) {
      const { status } = extractApiError(err)
      if (status === 400) toast.error('Configure a publish target first')
      else toastApiError(err)
    } finally {
      setPublishBusy(false)
    }
  }

  const validBadge = targetData?.valid
    ? 'bg-green-50 text-green-800 ring-green-200 dark:bg-green-950 dark:text-green-300 dark:ring-green-800'
    : 'bg-red-50 text-red-800 ring-red-200 dark:bg-red-950 dark:text-red-300 dark:ring-red-800'

  return (
    <div className="flex flex-col gap-4 max-w-3xl">
      <div className="flex items-center gap-2">
        <UploadCloud className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        <h1 className="text-sm font-semibold">Publish</h1>
        {hasActiveRun && (
          <span className="inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-800">
            Run active — disabled
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground -mt-2">
        Push the curated catalog to the site target. Choose what to publish, preview with a dry run, then apply.
      </p>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Publish to site</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6 max-w-2xl">

            {/* Target info */}
            <div>
              <SectionLabel>Publish target</SectionLabel>

              {targetLoading && <Skeleton className="h-8 w-full" />}
              {targetError && (
                <div className="flex items-center gap-2 text-destructive text-sm">
                  <AlertCircle className="h-4 w-4 flex-none" />
                  <span>Failed to load target: {(targetError as Error).message}</span>
                </div>
              )}

              {targetData && (
                <div className="space-y-2 mb-3">
                  <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs">
                    <span>
                      <span className="text-muted-foreground">Detected: </span>
                      <span className="font-mono">{targetData.detected ?? '—'}</span>
                    </span>
                    <span>
                      <span className="text-muted-foreground">Configured: </span>
                      <span className="font-mono">{targetData.configured ?? '—'}</span>
                    </span>
                    <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset ${validBadge}`}>
                      {targetData.valid ? 'Valid' : 'Invalid'}
                    </span>
                  </div>
                </div>
              )}

              {/* Set target */}
              <div className="flex gap-2 items-center">
                <Input
                  type="text"
                  value={targetInput}
                  onChange={e => setTargetInput(e.target.value)}
                  placeholder="/path/to/ndif-website"
                  className="h-7 text-xs font-mono flex-1"
                  disabled={settingTarget}
                  onKeyDown={e => { if (e.key === 'Enter') handleSetTarget() }}
                />
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleSetTarget}
                  disabled={settingTarget || !targetInput.trim()}
                  className="h-7 text-xs shrink-0"
                >
                  {settingTarget ? 'Setting…' : 'Set target'}
                </Button>
              </div>
            </div>

            {/* What to publish */}
            <div>
              <SectionLabel>What to publish</SectionLabel>
              <div className="flex flex-wrap gap-4 text-xs">
                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                  <input type="checkbox" checked={pubPapers} onChange={e => setPubPapers(e.target.checked)} />
                  Publish papers
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                  <input type="checkbox" checked={pubRepos} onChange={e => setPubRepos(e.target.checked)} />
                  Publish repos
                </label>
              </div>
              {noScope && (
                <p className="text-xs text-muted-foreground mt-1.5">Select papers and/or repos to enable publishing.</p>
              )}
            </div>

            {/* Dry run + Publish */}
            <div>
              <SectionLabel>Actions</SectionLabel>
              <div className="flex gap-2 flex-wrap">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleDryRun}
                  disabled={hasActiveRun || dryRunBusy || publishBusy || noScope}
                  className="h-7 text-xs gap-1"
                >
                  {dryRunBusy ? 'Running…' : 'Dry run'}
                </Button>

                <Button
                  size="sm"
                  onClick={() => setPublishConfirmOpen(true)}
                  disabled={hasActiveRun || publishBusy || dryRunBusy || noScope}
                  className="h-7 text-xs gap-1"
                >
                  <Upload className="h-3 w-3" />
                  {publishBusy ? 'Publishing…' : 'Publish'}
                </Button>

                {hasActiveRun && (
                  <span className="text-xs text-muted-foreground self-center">
                    Disabled during active run
                  </span>
                )}
              </div>
            </div>

            {/* Dry-run result */}
            {dryRunResult && <DiffSummary result={dryRunResult} />}

            {/* Publish result */}
            {publishResult && (
              <div className="rounded-md border bg-muted/30 p-3 text-xs space-y-2">
                <p className="font-medium text-foreground">Publish complete</p>
                {publishResult.summary && (
                  <p className="text-muted-foreground">
                    {publishResult.summary.files_written.length} file(s) written,{' '}
                    {publishResult.summary.images_copied} image(s) copied,{' '}
                    {publishResult.summary.images_overwritten} overwritten,{' '}
                    {publishResult.summary.images_unchanged} unchanged
                  </p>
                )}
                {publishResult.build_hint && (
                  <div className="rounded bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 p-2">
                    <p className="font-semibold text-amber-800 dark:text-amber-300 mb-0.5">Next step</p>
                    <p className="font-mono text-amber-700 dark:text-amber-400">{publishResult.build_hint}</p>
                  </div>
                )}
              </div>
            )}

            {/* Publish confirm dialog */}
            <AlertDialog open={publishConfirmOpen} onOpenChange={setPublishConfirmOpen}>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Publish to site?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will write the selected data ({[pubPapers && 'papers', pubRepos && 'repos'].filter(Boolean).join(' + ')}) to the
                    configured target. Run a dry run first to preview changes. The site will need a build
                    afterwards.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel onClick={() => setPublishConfirmOpen(false)}>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handlePublish}>Publish</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Export</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between gap-4 max-w-2xl">
            <p className="text-xs text-muted-foreground">
              Download the full catalog as a multi-sheet workbook (Papers, Pending, Discarded, GitHub).
            </p>
            <Button onClick={handleExport} disabled={exporting} variant="outline" size="sm" className="h-7 text-xs gap-1 shrink-0">
              <Download className="h-3 w-3" />
              {exporting ? 'Exporting…' : 'Export .xlsx'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
