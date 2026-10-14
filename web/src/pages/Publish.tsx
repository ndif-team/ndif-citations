import { useState } from 'react'
import { Upload, UploadCloud, Download } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  AlertDialog, AlertDialogContent, AlertDialogHeader, AlertDialogTitle,
  AlertDialogDescription, AlertDialogFooter, AlertDialogAction, AlertDialogCancel,
} from '@/components/ui/alert-dialog'
import { useActiveRun } from '@/api/hooks'
import { previewPublish, pushCatalog, exportXlsx } from '@/api/client'
import { toast } from 'sonner'
import { SectionLabel, toastApiError, extractApiError } from '@/lib/apiError'
import type { PublishPreviewResponse, PushCatalogResponse } from '@/api/types'

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

function DiffSummary({ result }: { result: PublishPreviewResponse }) {
  const { papers, repos, images } = result
  const len = (a: unknown[] | undefined) => (Array.isArray(a) ? a.length : 0)
  const total =
    len(papers?.added) + len(papers?.changed) + len(papers?.removed) +
    len(repos?.added) + len(repos?.changed) + len(repos?.removed) +
    len(images?.new) + len(images?.changed)
  return (
    <div className="rounded-md border bg-muted/30 p-3 text-xs space-y-2">
      <p className="font-medium text-foreground">
        Changes since the last push {total === 0 && <span className="font-normal text-muted-foreground">— nothing to push</span>}
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

  const [previewBusy, setPreviewBusy] = useState(false)
  const [preview, setPreview] = useState<PublishPreviewResponse | null>(null)
  const [pushBusy, setPushBusy] = useState(false)
  const [pushResult, setPushResult] = useState<PushCatalogResponse | null>(null)
  const [conflict, setConflict] = useState<string | null>(null)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [exporting, setExporting] = useState(false)

  async function handleExport() {
    setExporting(true)
    try { await exportXlsx(); toast.success('Exported .xlsx') }
    catch (e) { toast.error((e as Error).message) }
    finally { setExporting(false) }
  }

  async function handlePreview() {
    setPreviewBusy(true); setPreview(null)
    try { setPreview(await previewPublish()) }
    catch (err) { toastApiError(err) }
    finally { setPreviewBusy(false) }
  }

  async function handlePush() {
    setConfirmOpen(false); setPushBusy(true); setPushResult(null); setConflict(null)
    try {
      const res = await pushCatalog()
      setPushResult(res)
      toast.success(res.pushed ? `Pushed ${res.commit}` : 'Nothing to push')
    } catch (err) {
      const { status, message } = extractApiError(err)
      if (status === 409) setConflict(message)
      else toastApiError(err)
    } finally { setPushBusy(false) }
  }

  return (
    <div className="flex flex-col gap-4 max-w-3xl">
      <div className="flex items-center gap-2">
        <UploadCloud className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        <h1 className="text-sm font-semibold">Publish</h1>
      </div>
      <p className="text-xs text-muted-foreground -mt-2">
        Push your catalog changes to GitHub. ndif.us picks them up on its next daily build (or when the site
        workflow is run manually). The weekly cloud run publishes automatically — use this after local curation.
      </p>

      <Card>
        <CardHeader><CardTitle className="text-sm">Push catalog</CardTitle></CardHeader>
        <CardContent>
          <div className="space-y-4 max-w-2xl">
            <div>
              <SectionLabel>Actions</SectionLabel>
              <div className="flex gap-2 flex-wrap">
                <Button size="sm" variant="outline" onClick={handlePreview}
                        disabled={hasActiveRun || previewBusy || pushBusy} className="h-7 text-xs">
                  {previewBusy ? 'Checking…' : 'Preview changes'}
                </Button>
                <Button size="sm" onClick={() => setConfirmOpen(true)}
                        disabled={hasActiveRun || pushBusy || previewBusy} className="h-7 text-xs gap-1">
                  <Upload className="h-3 w-3" />
                  {pushBusy ? 'Pushing…' : 'Push catalog'}
                </Button>
                {hasActiveRun && <span className="text-xs text-muted-foreground self-center">Disabled during active run</span>}
              </div>
            </div>

            {preview && <DiffSummary result={preview} />}

            {pushResult && (
              <div className="rounded-md border bg-muted/30 p-3 text-xs">
                <p className="font-medium">{pushResult.pushed ? `Pushed ${pushResult.commit}` : 'Nothing to push'}</p>
                <p className="text-muted-foreground mt-1">{pushResult.message}</p>
              </div>
            )}

            {conflict && (
              <div className="rounded border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-300">
                <p className="font-semibold mb-1">Not pushed</p>
                <p className="font-mono break-words">{conflict}</p>
              </div>
            )}

            <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Push the catalog?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This commits your local catalog changes and pushes them to ndif-citations on GitHub.
                    Verified papers go live on ndif.us at its next build. Preview first if unsure.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel onClick={() => setConfirmOpen(false)}>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handlePush}>Push</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-sm">Export</CardTitle></CardHeader>
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
