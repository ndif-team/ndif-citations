import { useState, useMemo } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  type ColumnDef,
} from '@tanstack/react-table'
import { Search, AlertCircle, GitBranch, ExternalLink, RefreshCw } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { startRepoRefresh } from '@/api/client' 
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip'
import { useRepos, useActiveRun } from '@/api/hooks'
import { useDebounce } from '@/hooks/useDebounce'
import { categoryBadge, repoTypeBadge, repoTypeLabel } from '@/lib/tokens'
import { truncate } from '@/lib/utils'
import { RepoSheet } from '@/components/repos/RepoSheet'
import type { RepoRow, RepoType, RepoSortOption } from '@/api/types'

const REPO_TYPES: { label: string; value: '' | RepoType }[] = [
  { label: 'All', value: '' },
  { label: 'Research', value: 'research' },
  { label: 'Course', value: 'course' },
  { label: 'Experiment', value: 'experiment' },
]

const SORT_OPTIONS: { label: string; value: RepoSortOption }[] = [
  { label: 'Stars ↓', value: 'stars_desc' },
  { label: 'Recent', value: 'recent' },
  { label: 'Added ↓', value: 'added' },
  { label: 'Name', value: 'name' },
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
// Repos page
// ---------------------------------------------------------------------------

export function Repos() {
  const [repoType, setRepoType] = useState<'' | RepoType>('')
  const [searchInput, setSearchInput] = useState('')
  const [sort, setSort] = useState<RepoSortOption>('stars_desc')
  const [selectedOwner, setSelectedOwner] = useState<string | null>(null)
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null)

  // useActiveRun kept here so the cache is warm for RepoSheet, and to
  // disable the refresh button while any run owns the pipeline.
  const { data: activeRunData } = useActiveRun()
  const hasActiveRun = !!activeRunData?.active
  const navigate = useNavigate()
  const [refreshing, setRefreshing] = useState(false)

  async function handleRefresh() {
    setRefreshing(true)
    try {
      await startRepoRefresh()
      toast.success('Repo refresh started — stats updating in the background')
      navigate('/runs')
    } catch (err) {
      const status = (err as { status?: number }).status
      if (status === 409) toast.error('A run is already active — wait for it to finish')
      else toast.error(`Failed to start refresh: ${(err as Error).message}`)
    } finally {
      setRefreshing(false)
    }
  }

  const q = useDebounce(searchInput, 300)

  const { data: repos, isLoading, error } = useRepos({
    repo_type: repoType || undefined,
    q: q || undefined,
    sort,
  })

  const columns = useMemo<ColumnDef<RepoRow>[]>(
    () => [
      {
        id: 'owner_repo',
        header: 'Owner / Repo',
        size: 220,
        cell: ({ row }) => {
          const r = row.original
          return (
            <span className="flex items-center gap-1 font-mono text-xs font-medium">
              {r.manual_override && (
                <span className="w-1.5 h-1.5 rounded-full bg-orange-400 flex-none" title="Manual override" aria-label="Manual override" />
              )}
              {r.url ? (
                <a
                  href={r.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:underline text-primary flex items-center gap-1"
                  onClick={e => e.stopPropagation()}
                >
                  {r.owner}/{r.repo}
                  <ExternalLink className="h-3 w-3 opacity-50" aria-hidden="true" />
                </a>
              ) : (
                <span>{r.owner}/{r.repo}</span>
              )}
            </span>
          )
        },
      },
      {
        id: 'description',
        header: 'Description',
        accessorKey: 'description',
        size: 260,
        cell: ({ getValue }) => {
          const desc = getValue<string | null>()
          if (!desc) return <span className="text-xs text-muted-foreground">—</span>
          const truncated = truncate(desc, 80)
          if (desc === truncated) {
            return <span className="text-xs text-muted-foreground">{desc}</span>
          }
          return (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-xs text-muted-foreground cursor-default">{truncated}</span>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-sm">{desc}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )
        },
      },
      {
        id: 'stars',
        header: 'Stars',
        accessorKey: 'stars',
        size: 64,
        cell: ({ getValue }) => {
          const v = getValue<number | null>()
          return (
            <span className="text-xs tabular-nums">
              {v !== null ? v.toLocaleString() : '—'}
            </span>
          )
        },
      },
      {
        id: 'language',
        header: 'Language',
        accessorKey: 'language',
        size: 96,
        cell: ({ getValue }) => (
          <span className="text-xs text-muted-foreground">
            {getValue<string | null>() ?? '—'}
          </span>
        ),
      },
      {
        id: 'repo_type',
        header: 'Type',
        accessorKey: 'repo_type',
        size: 92,
        cell: ({ getValue }) => {
          const rt = getValue<RepoType>()
          return <span className={repoTypeBadge(rt)}>{repoTypeLabel(rt)}</span>
        },
      },
      {
        id: 'category',
        header: 'Category',
        accessorKey: 'category',
        size: 110,
        cell: ({ getValue }) => {
          const cat = getValue<string>()
          return (
            <span className={categoryBadge(cat as Parameters<typeof categoryBadge>[0])}>
              {cat}
            </span>
          )
        },
      },
      {
        id: 'linked_paper',
        header: '',
        accessorKey: 'linked_paper_url',
        size: 36,
        cell: ({ getValue }) => {
          const url = getValue<string | null>()
          if (!url) return null
          return (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={e => e.stopPropagation()}
                    className="inline-flex items-center text-muted-foreground hover:text-primary"
                    aria-label="Linked paper"
                  >
                    <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                  </a>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-xs break-all">{url}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )
        },
      },
    ],
    [],
  )

  const table = useReactTable({
    data: repos ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
  })

  function handleRowClick(row: RepoRow) {
    setSelectedOwner(row.owner)
    setSelectedRepo(row.repo)
  }

  return (
    <div className="flex flex-col gap-3 h-full max-h-[calc(100vh-96px)]">
      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Repo type pill tabs */}
        <div
          className="inline-flex h-8 items-center rounded-lg bg-muted p-1 gap-0.5"
          role="tablist"
          aria-label="Filter by repo type"
        >
          {REPO_TYPES.map(({ label, value }) => (
            <button
              key={value === '' ? '__all__' : value}
              role="tab"
              aria-selected={repoType === value}
              onClick={() => setRepoType(value)}
              className={[
                'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1 text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                repoType === value
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
            placeholder="Search repos…"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="pl-8 h-8 text-xs"
            aria-label="Search repos"
          />
        </div>

        {/* Sort */}
        <Select value={sort} onValueChange={(v) => setSort(v as RepoSortOption)}>
          <SelectTrigger className="w-28 h-8 text-xs" aria-label="Sort repos">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SORT_OPTIONS.map(({ label, value }) => (
              <SelectItem key={value} value={value}>{label}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Refresh stats — repos-only run: re-fetch GitHub stats + staleness, no scrape */}
        <Button
          size="sm"
          variant="outline"
          className="h-8 gap-1.5 text-xs"
          onClick={handleRefresh}
          disabled={refreshing || hasActiveRun}
          title="Re-fetch GitHub stats for all catalog repos (404/renamed/archived removed; no new repos discovered)"
        >
          <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
          Refresh stats
        </Button>

        {/* Count */}
        {!isLoading && repos && (
          <span className="text-xs text-muted-foreground tabular-nums ml-auto">
            {repos.length} repos
          </span>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-destructive text-sm">
          <AlertCircle className="h-4 w-4 flex-none" aria-hidden="true" />
          <span>Failed to load repos: {(error as Error).message}</span>
        </div>
      )}

      {/* Table */}
      <div className="flex-1 flex flex-col overflow-hidden rounded-md border bg-card">
        <div className="flex-1 overflow-auto">
          {isLoading ? (
            <div className="p-3">
              <TableSkeleton />
            </div>
          ) : (
            <table
              className="sticky-header w-full text-left border-collapse"
              aria-label="Repos table"
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
                      <GitBranch className="h-8 w-8 mx-auto mb-2 opacity-30" aria-hidden="true" />
                      No repos match
                    </td>
                  </tr>
                ) : (
                  table.getRowModel().rows.map(row => (
                    <tr
                      key={row.id}
                      onClick={() => handleRowClick(row.original)}
                      className="border-t border-border/50 hover:bg-muted/50 cursor-pointer transition-colors"
                      tabIndex={0}
                      role="button"
                      aria-label={`View details for ${row.original.owner}/${row.original.repo}`}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          handleRowClick(row.original)
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
      </div>

      {/* Repo detail drawer */}
      <RepoSheet
        owner={selectedOwner}
        repo={selectedRepo}
        onClose={() => {
          setSelectedOwner(null)
          setSelectedRepo(null)
        }}
      />
    </div>
  )
}
