import { useState, useCallback, useRef, useEffect } from 'react'
import type { KeyboardEvent } from 'react'
import { AlertCircle, X, Plus, Settings as SettingsIcon, Building2, Upload, KeyRound, CheckCircle2, XCircle } from 'lucide-react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
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
import { useSettings, useVenues, usePublishTarget, useActiveRun } from '@/api/hooks'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { putSettings, putVenues, putPublishTarget, runPublish, getKeys, putKeys, testKey } from '@/api/client'
import { toast } from 'sonner'
import { Badge } from '@/components/ui/badge'
import type { SettingsResponse, VenueEntry, VenueType, PublishDryRunResponse, PublishResponse } from '@/api/types'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const VENUE_TYPE_OPTIONS: { value: VenueType; label: string }[] = [
  { value: 'conference', label: 'Conference' },
  { value: 'workshop',   label: 'Workshop' },
  { value: 'journal',    label: 'Journal' },
  { value: 'preprint',   label: 'Preprint' },
]

const VENUE_TYPE_BADGE: Record<VenueType, string> = {
  conference: 'bg-blue-50 text-blue-800 ring-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:ring-blue-800',
  workshop:   'bg-violet-50 text-violet-800 ring-violet-200 dark:bg-violet-950 dark:text-violet-300 dark:ring-violet-800',
  journal:    'bg-green-50 text-green-800 ring-green-200 dark:bg-green-950 dark:text-green-300 dark:ring-green-800',
  preprint:   'bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-800',
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusClass(status: number): string {
  if (status === 409) return 'A run is in progress — try again when it finishes'
  if (status === 422) return 'Validation error'
  if (status === 400) return 'Bad request'
  return 'Unexpected error'
}

function extractApiError(err: unknown): { status?: number; message: string } {
  const e = err as { status?: number; message?: string }
  return { status: e.status, message: e.message ?? String(err) }
}

function toastApiError(err: unknown, fallback?: string) {
  const { status, message } = extractApiError(err)
  const prefix = statusClass(status ?? 0)
  if (status === 409) {
    toast.error(prefix)
  } else if (status === 422) {
    toast.error(`${prefix}: ${message}`)
  } else if (status === 400) {
    toast.error(fallback ?? prefix)
  } else {
    toast.error(message)
  }
}

// ---------------------------------------------------------------------------
// TagEditor — chip-based string[] editor
// ---------------------------------------------------------------------------

interface TagEditorProps {
  values: string[]
  onChange: (next: string[]) => void
  placeholder?: string
  disabled?: boolean
}

function TagEditor({ values, onChange, placeholder = 'Add…', disabled = false }: TagEditorProps) {
  const [input, setInput] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  function addTag(raw: string) {
    const val = raw.trim()
    if (!val || values.includes(val)) {
      setInput('')
      return
    }
    onChange([...values, val])
    setInput('')
  }

  function removeTag(idx: number) {
    onChange(values.filter((_, i) => i !== idx))
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault()
      addTag(input)
    } else if (e.key === 'Backspace' && input === '' && values.length > 0) {
      removeTag(values.length - 1)
    }
  }

  return (
    <div
      className="flex flex-wrap gap-1 p-1.5 rounded-md border border-input bg-background min-h-[36px] cursor-text"
      onClick={() => inputRef.current?.focus()}
    >
      {values.map((v, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-0.5 rounded-sm bg-muted px-1.5 py-0.5 text-xs font-mono"
        >
          {v}
          {!disabled && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); removeTag(i) }}
              className="ml-0.5 opacity-50 hover:opacity-100 focus:outline-none"
              aria-label={`Remove ${v}`}
            >
              <X className="h-2.5 w-2.5" />
            </button>
          )}
        </span>
      ))}
      {!disabled && (
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => { if (input.trim()) addTag(input) }}
          placeholder={values.length === 0 ? placeholder : ''}
          className="flex-1 min-w-[120px] bg-transparent text-xs outline-none placeholder:text-muted-foreground px-0.5"
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section label helper
// ---------------------------------------------------------------------------

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider pt-2 pb-1 border-b mb-3">
      {children}
    </p>
  )
}

function FieldLabel({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) {
  return (
    <label htmlFor={htmlFor} className="text-xs font-medium text-muted-foreground uppercase tracking-wider block mb-1">
      {children}
    </label>
  )
}

// ---------------------------------------------------------------------------
// Section 1 — Settings form
// ---------------------------------------------------------------------------

type SettingsDraft = {
  min_paper_year: number
  shared_paper_threshold: number
  llm_model: string
  llm_base_url: string
  llm_rate_limit_sleep: number
  s2_rate_limit_sleep: number
  github_rate_limit_sleep: number
  excluded_github_repos: string[]
  known_course_sources: string[]
  course_name_patterns: string[]
  ndif_keywords: string[]
  ndif_readme_keywords_regex: string[]
  ndif_readme_keywords_substr: string[]
  ndif_readme_negative_patterns: string[]
}

function settingsToLocal(s: SettingsResponse): SettingsDraft {
  return {
    min_paper_year: s.min_paper_year,
    shared_paper_threshold: s.shared_paper_threshold,
    llm_model: s.llm_model ?? '',
    llm_base_url: s.llm_base_url ?? '',
    llm_rate_limit_sleep: s.llm_rate_limit_sleep,
    s2_rate_limit_sleep: s.s2_rate_limit_sleep,
    github_rate_limit_sleep: s.github_rate_limit_sleep,
    excluded_github_repos: Array.isArray(s.excluded_github_repos) ? s.excluded_github_repos : [],
    known_course_sources: Array.isArray(s.known_course_sources) ? s.known_course_sources : [],
    course_name_patterns: Array.isArray(s.course_name_patterns) ? s.course_name_patterns : [],
    ndif_keywords: Array.isArray(s.ndif_keywords) ? s.ndif_keywords : [],
    ndif_readme_keywords_regex: Array.isArray(s.ndif_readme_keywords_regex) ? s.ndif_readme_keywords_regex : [],
    ndif_readme_keywords_substr: Array.isArray(s.ndif_readme_keywords_substr) ? s.ndif_readme_keywords_substr : [],
    ndif_readme_negative_patterns: Array.isArray(s.ndif_readme_negative_patterns) ? s.ndif_readme_negative_patterns : [],
  }
}

function buildPatch(draft: SettingsDraft, original: SettingsResponse): Partial<SettingsResponse> {
  const patch: Partial<SettingsResponse> = {}
  const d = draft

  if (d.min_paper_year !== original.min_paper_year) patch.min_paper_year = d.min_paper_year
  if (d.shared_paper_threshold !== original.shared_paper_threshold) patch.shared_paper_threshold = d.shared_paper_threshold
  if (d.llm_model !== (original.llm_model ?? '')) patch.llm_model = d.llm_model
  if (d.llm_base_url !== (original.llm_base_url ?? '')) patch.llm_base_url = d.llm_base_url
  if (d.llm_rate_limit_sleep !== original.llm_rate_limit_sleep) patch.llm_rate_limit_sleep = d.llm_rate_limit_sleep
  if (d.s2_rate_limit_sleep !== original.s2_rate_limit_sleep) patch.s2_rate_limit_sleep = d.s2_rate_limit_sleep
  if (d.github_rate_limit_sleep !== original.github_rate_limit_sleep) patch.github_rate_limit_sleep = d.github_rate_limit_sleep

  const listKeys: (keyof SettingsDraft & keyof SettingsResponse)[] = [
    'excluded_github_repos',
    'known_course_sources',
    'course_name_patterns',
    'ndif_keywords',
    'ndif_readme_keywords_regex',
    'ndif_readme_keywords_substr',
    'ndif_readme_negative_patterns',
  ]
  for (const key of listKeys) {
    const dArr = d[key] as string[]
    const oArr = (Array.isArray(original[key]) ? original[key] : []) as string[]
    if (JSON.stringify(dArr) !== JSON.stringify(oArr)) {
      ;(patch as Record<string, unknown>)[key] = dArr
    }
  }
  return patch
}

function SettingsSection({ hasActiveRun }: { hasActiveRun: boolean }) {
  const { data, isLoading, error } = useSettings()
  const qc = useQueryClient()
  const [draft, setDraft] = useState<SettingsDraft | null>(null)
  const [saving, setSaving] = useState(false)

  // Sync draft when server data arrives (first time or after reset)
  const initializedRef = useRef(false)
  useEffect(() => {
    if (data && !initializedRef.current) {
      setDraft(settingsToLocal(data))
      initializedRef.current = true
    }
  }, [data])

  function reset() {
    if (data) {
      setDraft(settingsToLocal(data))
    }
  }

  const handleSave = useCallback(async () => {
    if (!draft || !data) return
    const patch = buildPatch(draft, data)
    if (Object.keys(patch).length === 0) {
      toast.info('No changes to save')
      return
    }
    setSaving(true)
    try {
      const updated = await putSettings(patch)
      qc.setQueryData(['settings'], updated)
      initializedRef.current = false // allow re-sync on next render
      toast.success('Settings saved')
    } catch (err) {
      toastApiError(err)
    } finally {
      setSaving(false)
    }
  }, [draft, data, qc])

  if (isLoading) {
    return (
      <div className="space-y-3 max-w-2xl">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-8 w-full" />
        ))}
      </div>
    )
  }

  if (error || !draft) {
    return (
      <div className="flex items-center gap-2 text-destructive text-sm">
        <AlertCircle className="h-4 w-4 flex-none" />
        <span>Failed to load settings{error ? `: ${(error as Error).message}` : ''}</span>
      </div>
    )
  }

  function setNum(key: keyof SettingsDraft, raw: string) {
    const n = parseFloat(raw)
    if (!isNaN(n)) setDraft(d => d ? { ...d, [key]: n } : d)
    else setDraft(d => d ? { ...d, [key]: raw as unknown as number } : d)
  }

  function setStr(key: keyof SettingsDraft, val: string) {
    setDraft(d => d ? { ...d, [key]: val } : d)
  }

  function setList(key: keyof SettingsDraft, val: string[]) {
    setDraft(d => d ? { ...d, [key]: val } : d)
  }

  const disabled = hasActiveRun || saving

  return (
    <div className="space-y-6 max-w-2xl">
      {/* Discovery */}
      <div>
        <SectionLabel>Discovery</SectionLabel>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <FieldLabel>Min paper year</FieldLabel>
            <Input
              type="number"
              value={draft.min_paper_year}
              onChange={e => setNum('min_paper_year', e.target.value)}
              disabled={disabled}
              className="h-7 text-xs"
            />
          </div>
          <div>
            <FieldLabel>Shared paper threshold</FieldLabel>
            <Input
              type="number"
              value={draft.shared_paper_threshold}
              onChange={e => setNum('shared_paper_threshold', e.target.value)}
              disabled={disabled}
              className="h-7 text-xs"
            />
          </div>
        </div>
      </div>

      {/* LLM */}
      <div>
        <SectionLabel>LLM</SectionLabel>
        <div className="grid gap-3">
          <div>
            <FieldLabel>Model</FieldLabel>
            <Input
              type="text"
              value={draft.llm_model}
              onChange={e => setStr('llm_model', e.target.value)}
              disabled={disabled}
              className="h-7 text-xs font-mono"
              placeholder="e.g. gpt-4o"
            />
          </div>
          <div>
            <FieldLabel>Base URL</FieldLabel>
            <Input
              type="text"
              value={draft.llm_base_url}
              onChange={e => setStr('llm_base_url', e.target.value)}
              disabled={disabled}
              className="h-7 text-xs font-mono"
              placeholder="https://api.openai.com/v1"
            />
          </div>
          <div>
            <FieldLabel>Rate limit sleep (s)</FieldLabel>
            <Input
              type="number"
              step="0.1"
              value={draft.llm_rate_limit_sleep}
              onChange={e => setNum('llm_rate_limit_sleep', e.target.value)}
              disabled={disabled}
              className="h-7 text-xs w-32"
            />
          </div>
        </div>
      </div>

      {/* Rate limits */}
      <div>
        <SectionLabel>Rate limits</SectionLabel>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <FieldLabel>S2 sleep (s)</FieldLabel>
            <Input
              type="number"
              step="0.1"
              value={draft.s2_rate_limit_sleep}
              onChange={e => setNum('s2_rate_limit_sleep', e.target.value)}
              disabled={disabled}
              className="h-7 text-xs"
            />
          </div>
          <div>
            <FieldLabel>GitHub sleep (s)</FieldLabel>
            <Input
              type="number"
              step="0.1"
              value={draft.github_rate_limit_sleep}
              onChange={e => setNum('github_rate_limit_sleep', e.target.value)}
              disabled={disabled}
              className="h-7 text-xs"
            />
          </div>
        </div>
      </div>

      {/* Lists */}
      <div>
        <SectionLabel>Lists</SectionLabel>
        <div className="grid gap-4">
          {([
            { key: 'excluded_github_repos',       label: 'Excluded GitHub repos',          ph: 'owner/repo' },
            { key: 'known_course_sources',         label: 'Known course sources',            ph: 'source name' },
            { key: 'course_name_patterns',         label: 'Course name patterns',            ph: 'pattern' },
            { key: 'ndif_keywords',                label: 'NDIF keywords',                   ph: 'keyword' },
            { key: 'ndif_readme_keywords_regex',   label: 'README keywords (regex)',         ph: 'regex' },
            { key: 'ndif_readme_keywords_substr',  label: 'README keywords (substring)',     ph: 'substring' },
            { key: 'ndif_readme_negative_patterns',label: 'README negative patterns',        ph: 'pattern' },
          ] as { key: keyof SettingsDraft; label: string; ph: string }[]).map(({ key, label, ph }) => (
            <div key={key}>
              <FieldLabel>{label}</FieldLabel>
              <TagEditor
                values={(draft[key] as string[]) ?? []}
                onChange={val => setList(key, val)}
                placeholder={ph}
                disabled={disabled}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2 pt-2 border-t">
        <Button
          size="sm"
          onClick={handleSave}
          disabled={disabled}
        >
          {saving ? 'Saving…' : 'Save changes'}
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={reset}
          disabled={saving}
        >
          Reset
        </Button>
        {hasActiveRun && (
          <span className="text-xs text-muted-foreground self-center ml-2">
            Disabled during active run
          </span>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 2 — Venue manager
// ---------------------------------------------------------------------------

interface VenueRow {
  canonical: string
  type: VenueType
  aliases: string[]
  parent: string
}

function venueMapToRows(venues: Record<string, VenueEntry>): VenueRow[] {
  return Object.entries(venues).map(([canonical, entry]) => ({
    canonical,
    type: entry.type,
    aliases: Array.isArray(entry.aliases) ? entry.aliases : [],
    parent: entry.parent ?? '',
  }))
}

function rowsToVenueMap(rows: VenueRow[]): Record<string, VenueEntry> {
  const out: Record<string, VenueEntry> = {}
  for (const row of rows) {
    if (!row.canonical.trim()) continue
    const entry: VenueEntry = { type: row.type }
    if (row.aliases.length > 0) entry.aliases = row.aliases
    if (row.parent.trim()) entry.parent = row.parent.trim()
    out[row.canonical.trim()] = entry
  }
  return out
}

function VenuesSection({ hasActiveRun }: { hasActiveRun: boolean }) {
  const { data, isLoading, error } = useVenues()
  const qc = useQueryClient()
  const [rows, setRows] = useState<VenueRow[]>([])
  const [saving, setSaving] = useState(false)

  const initializedRef = useRef(false)
  useEffect(() => {
    if (data && !initializedRef.current) {
      setRows(venueMapToRows(data.venues))
      initializedRef.current = true
    }
  }, [data])

  function addRow() {
    setRows(r => [...r, { canonical: '', type: 'conference', aliases: [], parent: '' }])
  }

  function removeRow(idx: number) {
    setRows(r => r.filter((_, i) => i !== idx))
  }

  function updateRow(idx: number, patch: Partial<VenueRow>) {
    setRows(r => r.map((row, i) => i === idx ? { ...row, ...patch } : row))
  }

  const handleSave = useCallback(async () => {
    const venueMap = rowsToVenueMap(rows)
    setSaving(true)
    try {
      const updated = await putVenues(venueMap)
      qc.setQueryData(['venues'], updated)
      initializedRef.current = false
      toast.success('Venues saved')
    } catch (err) {
      toastApiError(err)
    } finally {
      setSaving(false)
    }
  }, [rows, qc])

  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-9 w-full" />)}
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-destructive text-sm">
        <AlertCircle className="h-4 w-4 flex-none" />
        <span>Failed to load venues: {(error as Error).message}</span>
      </div>
    )
  }

  const disabled = hasActiveRun || saving

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-muted/50">
              <th className="px-3 py-2 text-left font-medium text-muted-foreground whitespace-nowrap w-44">Canonical name</th>
              <th className="px-3 py-2 text-left font-medium text-muted-foreground whitespace-nowrap w-32">Type</th>
              <th className="px-3 py-2 text-left font-medium text-muted-foreground">Aliases</th>
              <th className="px-3 py-2 text-left font-medium text-muted-foreground w-36">Parent</th>
              <th className="px-3 py-2 w-8" />
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx} className="border-t border-border/50">
                {/* Canonical */}
                <td className="px-2 py-1.5">
                  <Input
                    value={row.canonical}
                    onChange={e => updateRow(idx, { canonical: e.target.value })}
                    disabled={disabled}
                    className="h-6 text-xs font-mono"
                    placeholder="NeurIPS"
                  />
                </td>
                {/* Type — no empty option; always a real value */}
                <td className="px-2 py-1.5">
                  <Select
                    value={row.type}
                    onValueChange={v => updateRow(idx, { type: v as VenueType })}
                    disabled={disabled}
                  >
                    <SelectTrigger className="h-6 text-xs w-28">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {VENUE_TYPE_OPTIONS.map(opt => (
                        <SelectItem key={opt.value} value={opt.value} className="text-xs">
                          <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset ${VENUE_TYPE_BADGE[opt.value]}`}>
                            {opt.label}
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </td>
                {/* Aliases */}
                <td className="px-2 py-1.5">
                  <TagEditor
                    values={row.aliases}
                    onChange={val => updateRow(idx, { aliases: val })}
                    placeholder="alias"
                    disabled={disabled}
                  />
                </td>
                {/* Parent */}
                <td className="px-2 py-1.5">
                  <Input
                    value={row.parent}
                    onChange={e => updateRow(idx, { parent: e.target.value })}
                    disabled={disabled}
                    className="h-6 text-xs font-mono"
                    placeholder="—"
                  />
                </td>
                {/* Remove */}
                <td className="px-2 py-1.5 text-center">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-6 w-6"
                    onClick={() => removeRow(idx)}
                    disabled={disabled}
                    aria-label="Remove venue"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={5} className="text-center py-8 text-xs text-muted-foreground">
                  No venues configured. Click "Add venue" to add one.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={addRow}
          disabled={disabled}
          className="gap-1 h-7 text-xs"
        >
          <Plus className="h-3 w-3" />
          Add venue
        </Button>
        <Button
          size="sm"
          onClick={handleSave}
          disabled={disabled}
          className="h-7 text-xs"
        >
          {saving ? 'Saving…' : 'Save venues'}
        </Button>
        {hasActiveRun && (
          <span className="text-xs text-muted-foreground self-center ml-2">
            Disabled during active run
          </span>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 3 — Publish
// ---------------------------------------------------------------------------

function isDryRunResponse(r: PublishDryRunResponse | PublishResponse): r is PublishDryRunResponse {
  return (r as PublishDryRunResponse).papers !== undefined
}

function DiffSummary({ result }: { result: PublishDryRunResponse }) {
  const { papers, repos, images } = result
  // diff buckets are arrays of affected records — render their lengths as counts.
  const len = (a: unknown[] | undefined) => (Array.isArray(a) ? a.length : 0)
  return (
    <div className="rounded-md border bg-muted/30 p-3 text-xs space-y-1.5">
      <p className="font-medium text-foreground">Dry-run diff</p>
      <div className="space-y-0.5 text-muted-foreground">
        <p>Papers: <span className="text-green-700 dark:text-green-400">+{len(papers?.added)} added</span>, <span className="text-amber-700 dark:text-amber-400">~{len(papers?.changed)} changed</span>, <span className="text-red-700 dark:text-red-400">-{len(papers?.removed)} removed</span></p>
        <p>Repos: <span className="text-green-700 dark:text-green-400">+{len(repos?.added)} added</span>, <span className="text-amber-700 dark:text-amber-400">~{len(repos?.changed)} changed</span>, <span className="text-red-700 dark:text-red-400">-{len(repos?.removed)} removed</span></p>
        <p>Images: <span className="text-green-700 dark:text-green-400">{len(images?.new)} new</span>, <span className="text-amber-700 dark:text-amber-400">{len(images?.changed)} changed</span></p>
      </div>
    </div>
  )
}

function PublishSection({ hasActiveRun }: { hasActiveRun: boolean }) {
  const { data: targetData, isLoading: targetLoading, error: targetError } = usePublishTarget()
  const qc = useQueryClient()

  const [targetInput, setTargetInput] = useState('')
  const [settingTarget, setSettingTarget] = useState(false)

  const [dryRunBusy, setDryRunBusy] = useState(false)
  const [dryRunResult, setDryRunResult] = useState<PublishDryRunResponse | null>(null)

  const [publishBusy, setPublishBusy] = useState(false)
  const [publishResult, setPublishResult] = useState<PublishResponse | null>(null)
  const [publishConfirmOpen, setPublishConfirmOpen] = useState(false)

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
      const res = await runPublish(true)
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
      const res = await runPublish(false)
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
            placeholder="/path/to/ndif-web-beta/packages/ndif.us"
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

      {/* Dry run + Publish */}
      <div>
        <SectionLabel>Actions</SectionLabel>
        <div className="flex gap-2 flex-wrap">
          <Button
            size="sm"
            variant="outline"
            onClick={handleDryRun}
            disabled={hasActiveRun || dryRunBusy || publishBusy}
            className="h-7 text-xs gap-1"
          >
            {dryRunBusy ? 'Running…' : 'Dry run'}
          </Button>

          <Button
            size="sm"
            onClick={() => setPublishConfirmOpen(true)}
            disabled={hasActiveRun || publishBusy || dryRunBusy}
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
            <p className="text-muted-foreground">{publishResult.summary}</p>
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
              This will write <code>research-papers.json</code> and <code>github-repos.json</code> to the
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
  )
}

// ---------------------------------------------------------------------------
// Section 4 — API Keys (write-only)
// ---------------------------------------------------------------------------

type KeyName = 'LLM_API_KEY' | 'S2_API_KEY' | 'GITHUB_TOKEN' | 'SERPAPI_API_KEY'
type TestProvider = 'llm' | 'github' | 's2'

const KEY_META: { name: KeyName; label: string; provider?: TestProvider }[] = [
  { name: 'LLM_API_KEY',      label: 'LLM API Key',       provider: 'llm' },
  { name: 'S2_API_KEY',       label: 'S2 API Key',        provider: 's2' },
  { name: 'GITHUB_TOKEN',     label: 'GitHub Token',      provider: 'github' },
  { name: 'SERPAPI_API_KEY',  label: 'SerpAPI API Key' },
]

function ApiKeysSection({ hasActiveRun }: { hasActiveRun: boolean }) {
  const qc = useQueryClient()
  const { data: keys, isLoading, error } = useQuery({
    queryKey: ['settings', 'keys'],
    queryFn: getKeys,
    staleTime: 30_000,
  })

  // Draft: one input string per key (empty = keep existing)
  const [draft, setDraft] = useState<Record<KeyName, string>>({
    LLM_API_KEY: '',
    S2_API_KEY: '',
    GITHUB_TOKEN: '',
    SERPAPI_API_KEY: '',
  })

  // Per-key test result
  const [testResults, setTestResults] = useState<Record<string, { ok: boolean; detail: string } | null>>({})
  const [testingKey, setTestingKey] = useState<KeyName | null>(null)
  const [saving, setSaving] = useState(false)

  function setKeyDraft(name: KeyName, value: string) {
    setDraft(d => ({ ...d, [name]: value }))
    // Clear test result for this key when user types
    setTestResults(r => ({ ...r, [name]: null }))
  }

  async function handleTest(name: KeyName, provider: TestProvider) {
    setTestingKey(name)
    setTestResults(r => ({ ...r, [name]: null }))
    try {
      const result = await testKey(provider)
      setTestResults(r => ({ ...r, [name]: result }))
    } catch (err) {
      toastApiError(err, 'Test request failed')
    } finally {
      setTestingKey(null)
    }
  }

  async function handleSave() {
    const changes: Record<string, string> = {}
    for (const { name } of KEY_META) {
      const v = draft[name].trim()
      if (v) changes[name] = v
    }
    if (Object.keys(changes).length === 0) {
      toast.info('No keys to save — enter a value in at least one field')
      return
    }
    setSaving(true)
    try {
      const updated = await putKeys(changes)
      qc.setQueryData(['settings', 'keys'], updated)
      // Refresh the run-start preflight gate so a newly-saved key unblocks runs immediately
      qc.invalidateQueries({ queryKey: ['preflight'] })
      // Clear inputs
      setDraft({ LLM_API_KEY: '', S2_API_KEY: '', GITHUB_TOKEN: '', SERPAPI_API_KEY: '' })
      setTestResults({})
      toast.success('API keys saved')
    } catch (err) {
      toastApiError(err)
    } finally {
      setSaving(false)
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-3 max-w-2xl">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-12 bg-muted animate-pulse rounded-md" />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-destructive text-sm">
        <AlertCircle className="h-4 w-4 flex-none" />
        <span>Failed to load key status: {(error as Error).message}</span>
      </div>
    )
  }

  const disabled = hasActiveRun || saving

  return (
    <div className="space-y-5 max-w-2xl">
      <SectionLabel>Secrets</SectionLabel>
      <p className="text-xs text-muted-foreground -mt-2">
        Values are write-only — leave a field blank to keep the existing key.
      </p>

      <div className="space-y-4">
        {KEY_META.map(({ name, label, provider }) => {
          const configured = keys?.[name]?.configured ?? false
          const testResult = testResults[name] ?? null
          const isTesting = testingKey === name

          return (
            <div key={name} className="grid gap-1.5">
              <FieldLabel htmlFor={`apikey-${name}`}>{label}</FieldLabel>
              <div className="flex items-center gap-2 flex-wrap">
                {/* Configured badge */}
                {configured ? (
                  <Badge className="shrink-0 bg-green-50 text-green-800 ring-green-200 dark:bg-green-950 dark:text-green-300 dark:ring-green-800 gap-1">
                    <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
                    Configured
                  </Badge>
                ) : (
                  <Badge variant="outline" className="shrink-0 text-muted-foreground gap-1">
                    <XCircle className="h-3 w-3" aria-hidden="true" />
                    Not set
                  </Badge>
                )}

                {/* Password input */}
                <Input
                  id={`apikey-${name}`}
                  type="password"
                  placeholder="leave blank to keep"
                  value={draft[name]}
                  onChange={e => setKeyDraft(name, e.target.value)}
                  disabled={disabled}
                  className="h-7 text-xs font-mono flex-1 min-w-[180px]"
                  autoComplete="new-password"
                />

                {/* Test button (not for SERPAPI) */}
                {provider && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs shrink-0"
                    disabled={disabled || isTesting}
                    onClick={() => handleTest(name, provider)}
                    title="Tests the saved key — save first to test a new value"
                  >
                    {isTesting ? 'Testing…' : 'Test'}
                  </Button>
                )}
              </div>

              {/* Test result inline */}
              {testResult && (
                <p className={`text-xs flex items-center gap-1 ${testResult.ok ? 'text-green-700 dark:text-green-400' : 'text-destructive'}`}>
                  {testResult.ok
                    ? <CheckCircle2 className="h-3 w-3 flex-none" aria-hidden="true" />
                    : <XCircle className="h-3 w-3 flex-none" aria-hidden="true" />}
                  {testResult.detail}
                </p>
              )}
            </div>
          )
        })}
      </div>

      <div className="flex gap-2 pt-2 border-t">
        <Button
          size="sm"
          onClick={handleSave}
          disabled={disabled}
        >
          {saving ? 'Saving…' : 'Save keys'}
        </Button>
        {hasActiveRun && (
          <span className="text-xs text-muted-foreground self-center ml-2">
            Disabled during active run
          </span>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Settings page (root)
// ---------------------------------------------------------------------------

export function Settings() {
  const { data: activeRunData } = useActiveRun()
  const hasActiveRun = !!(activeRunData?.active)

  return (
    <div className="flex flex-col gap-4 max-w-3xl">
      <div className="flex items-center gap-2">
        <SettingsIcon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        <h1 className="text-sm font-semibold">Settings</h1>
        {hasActiveRun && (
          <span className="inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-800">
            Run active — edits disabled
          </span>
        )}
      </div>

      <Tabs defaultValue="settings">
        <TabsList className="mb-2">
          <TabsTrigger value="settings" className="gap-1.5">
            <SettingsIcon className="h-3 w-3" aria-hidden="true" />
            Pipeline settings
          </TabsTrigger>
          <TabsTrigger value="venues" className="gap-1.5">
            <Building2 className="h-3 w-3" aria-hidden="true" />
            Venues
          </TabsTrigger>
          <TabsTrigger value="publish" className="gap-1.5">
            <Upload className="h-3 w-3" aria-hidden="true" />
            Publish
          </TabsTrigger>
          <TabsTrigger value="keys" className="gap-1.5">
            <KeyRound className="h-3 w-3" aria-hidden="true" />
            API Keys
          </TabsTrigger>
        </TabsList>

        <TabsContent value="settings">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Pipeline settings</CardTitle>
            </CardHeader>
            <CardContent>
              <SettingsSection hasActiveRun={hasActiveRun} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="venues">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Venue manager</CardTitle>
            </CardHeader>
            <CardContent>
              <VenuesSection hasActiveRun={hasActiveRun} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="publish">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Publish to site</CardTitle>
            </CardHeader>
            <CardContent>
              <PublishSection hasActiveRun={hasActiveRun} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="keys">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">API Keys</CardTitle>
            </CardHeader>
            <CardContent>
              <ApiKeysSection hasActiveRun={hasActiveRun} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
