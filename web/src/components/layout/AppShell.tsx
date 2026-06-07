import { Suspense } from 'react'
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Topbar } from './Topbar'

/** Shown while a lazily-loaded route chunk is fetched; keeps the shell in place. */
function RouteFallback() {
  return (
    <div
      className="flex h-full w-full items-center justify-center text-sm text-muted-foreground"
      role="status"
      aria-live="polite"
    >
      Loading…
    </div>
  )
}

export function AppShell() {
  return (
    <div className="flex h-full w-full overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Topbar />
        <main className="flex-1 overflow-auto p-4 md:p-6">
          <Suspense fallback={<RouteFallback />}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  )
}
