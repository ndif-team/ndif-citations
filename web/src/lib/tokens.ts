import type { Bucket, Category, ConfidenceBand, RunState } from '@/api/types'
import { cn } from './utils'

// Bucket semantic tokens — color is never the only signal
export function bucketBadge(bucket: Bucket): string {
  return cn(
    'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset',
    {
      verified: 'bg-green-50 text-green-800 ring-green-200 dark:bg-green-950 dark:text-green-300 dark:ring-green-800',
      pending: 'bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-800',
      discarded: 'bg-slate-100 text-slate-600 ring-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:ring-slate-700',
    }[bucket]
  )
}

export function bucketLabel(bucket: Bucket): string {
  return {
    verified: '✓ Verified',
    pending: '⏳ Pending',
    discarded: '✗ Discarded',
  }[bucket]
}

// Confidence band tokens
export function confidenceBadge(band: ConfidenceBand): string {
  return cn(
    'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset',
    {
      CERTAIN: 'bg-green-50 text-green-800 ring-green-200 dark:bg-green-950 dark:text-green-300 dark:ring-green-800',
      HIGH: 'bg-emerald-50 text-emerald-800 ring-emerald-200 dark:bg-emerald-950 dark:text-emerald-300 dark:ring-emerald-800',
      MEDIUM: 'bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-800',
      LOW: 'bg-orange-50 text-orange-800 ring-orange-200 dark:bg-orange-950 dark:text-orange-300 dark:ring-orange-800',
      NONE: 'bg-slate-100 text-slate-600 ring-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:ring-slate-700',
    }[band] ?? 'bg-slate-100 text-slate-600 ring-slate-200'
  )
}

// Category tokens
export function categoryBadge(category: Category): string {
  return cn(
    'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset',
    {
      uses_ndif: 'bg-blue-50 text-blue-800 ring-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:ring-blue-800',
      uses_nnsight: 'bg-violet-50 text-violet-800 ring-violet-200 dark:bg-violet-950 dark:text-violet-300 dark:ring-violet-800',
      referencing: 'bg-cyan-50 text-cyan-800 ring-cyan-200 dark:bg-cyan-950 dark:text-cyan-300 dark:ring-cyan-800',
      unclassified: 'bg-slate-100 text-slate-600 ring-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:ring-slate-700',
    }[category] ?? 'bg-slate-100 text-slate-600 ring-slate-200'
  )
}

export function categoryLabel(category: Category): string {
  return {
    uses_ndif: 'Uses NDIF',
    uses_nnsight: 'Uses NNsight',
    referencing: 'Referencing',
    unclassified: 'Unclassified',
  }[category] ?? category
}

// Run-state badge tokens
export function runStateBadge(state: RunState | string): string {
  return cn(
    'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset',
    {
      running: 'bg-blue-50 text-blue-800 ring-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:ring-blue-800',
      awaiting_review: 'bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-800',
      done: 'bg-green-50 text-green-800 ring-green-200 dark:bg-green-950 dark:text-green-300 dark:ring-green-800',
      error: 'bg-red-50 text-red-800 ring-red-200 dark:bg-red-950 dark:text-red-300 dark:ring-red-800',
      cancelled: 'bg-slate-100 text-slate-600 ring-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:ring-slate-700',
    }[state] ?? 'bg-slate-100 text-slate-600 ring-slate-200'
  )
}

export function runStateLabel(state: RunState | string): string {
  return {
    running: 'Running',
    awaiting_review: 'Awaiting review',
    done: 'Done',
    error: 'Error',
    cancelled: 'Cancelled',
  }[state] ?? state
}
