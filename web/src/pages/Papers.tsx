import { useState, useMemo } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  type ColumnDef,
} from '@tanstack/react-table'
import { Search, AlertCircle, FileText } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip'
import { usePapers } from '@/api/hooks'
import { useDebounce } from '@/hooks/useDebounce'
import { bucketBadge, confidenceBadge, categoryBadge, categoryLabel } from '@/lib/tokens'
import { formatAuthors, truncate } from '@/lib/utils'
import { PaperSheet } from '@/components/papers/PaperSheet'
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

function TableSkeleton() {
  return (
    <div className="space-y-1.5">
      {Array.from({ length: 12 }).map((_, i) => (
        <Skeleton key={i} className="h-9 w-full" />
      ))}
    </div>
  )
}

export function Papers() {
  const [bucket, setBucket] = useState<'' | Bucket>('')
  const [searchInput, setSearchInput] = useState('')
  const [sort, setSort] = useState<SortOption>('year_desc')
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const q = useDebounce(searchInput, 300)

  const { data: papers, isLoading, error } = usePapers({
    bucket: bucket || undefined,
    q: q || undefined,
    sort,
  })

  const columns = useMemo<ColumnDef<PaperRow>[]>(
    () => [
      {
        id: 'title',
        header: 'Title',
        accessorKey: 'title',
        size: 280,
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
    []
  )

  const table = useReactTable({
    data: papers ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
  })

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

        {/* Count */}
        {!isLoading && papers && (
          <span className="text-xs text-muted-foreground tabular-nums ml-auto">
            {papers.length} papers
          </span>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-destructive text-sm">
          <AlertCircle className="h-4 w-4 flex-none" aria-hidden="true" />
          <span>Failed to load papers: {(error as Error).message}</span>
        </div>
      )}

      {/* Table */}
      <div className="flex-1 overflow-auto rounded-md border bg-card">
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
                    className="border-t border-border/50 hover:bg-muted/50 cursor-pointer transition-colors"
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

      {/* Paper detail drawer */}
      <PaperSheet
        paperId={selectedId}
        onClose={() => setSelectedId(null)}
      />
    </div>
  )
}
