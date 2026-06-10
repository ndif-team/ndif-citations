import type { ReactNode } from 'react'
import { toast } from 'sonner'

/** Map an HTTP status to a short human-readable class of error. */
export function statusClass(status: number): string {
  if (status === 409) return 'A run is in progress — try again when it finishes'
  if (status === 422) return 'Validation error'
  if (status === 400) return 'Bad request'
  return 'Unexpected error'
}

export function extractApiError(err: unknown): { status?: number; message: string } {
  const e = err as { status?: number; message?: string }
  return { status: e.status, message: e.message ?? String(err) }
}

export function toastApiError(err: unknown, fallback?: string) {
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

/** Small uppercase section heading used across Settings/Publish forms. */
export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider pt-2 pb-1 border-b mb-3">
      {children}
    </p>
  )
}
