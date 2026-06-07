import { Link } from 'react-router-dom'
import { Moon, Sun, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/hooks/useTheme'
import { useActiveRun } from '@/api/hooks'
import { cn } from '@/lib/utils'

function RunIndicator() {
  const { data } = useActiveRun()
  const active = data?.active ?? null

  if (!active) return null

  const isAwaiting = active.state === 'awaiting_review'

  return (
    <Link
      to="/runs"
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset no-underline transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        isAwaiting
          ? 'bg-amber-50 text-amber-800 ring-amber-200 hover:bg-amber-100 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-800 dark:hover:bg-amber-900'
          : 'bg-blue-50 text-blue-800 ring-blue-200 hover:bg-blue-100 dark:bg-blue-950 dark:text-blue-300 dark:ring-blue-800 dark:hover:bg-blue-900'
      )}
      aria-label={isAwaiting ? 'Run awaiting review — click to review' : 'Run in progress — click to view'}
    >
      <Loader2
        className={cn(
          'h-3 w-3 flex-none animate-spin motion-reduce:animate-none',
          isAwaiting ? 'text-amber-600 dark:text-amber-400' : 'text-blue-500 dark:text-blue-400'
        )}
        aria-hidden="true"
      />
      {isAwaiting ? 'Awaiting review' : 'Run in progress'}
    </Link>
  )
}

export function Topbar() {
  const { theme, toggle } = useTheme()

  return (
    <header className="flex-none flex items-center justify-between px-4 h-12 border-b bg-card">
      <div className="flex items-center gap-2">
        <h1 className="text-sm font-semibold text-foreground tracking-tight">NDIF Citations</h1>
        <span className="hidden sm:inline text-xs text-muted-foreground">— curation dashboard</span>
      </div>
      <div className="flex items-center gap-2">
        <RunIndicator />
        <Button
          variant="ghost"
          size="icon"
          onClick={toggle}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </Button>
      </div>
    </header>
  )
}
