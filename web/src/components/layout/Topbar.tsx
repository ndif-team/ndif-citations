import { Moon, Sun } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/hooks/useTheme'

export function Topbar() {
  const { theme, toggle } = useTheme()

  return (
    <header className="flex-none flex items-center justify-between px-4 h-12 border-b bg-card">
      <div className="flex items-center gap-2">
        <h1 className="text-sm font-semibold text-foreground tracking-tight">NDIF Citations</h1>
        <span className="hidden sm:inline text-xs text-muted-foreground">— curation dashboard</span>
      </div>
      <div className="flex items-center gap-2">
        {/* Placeholder: run indicator slot */}
        <div id="run-indicator" />
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
