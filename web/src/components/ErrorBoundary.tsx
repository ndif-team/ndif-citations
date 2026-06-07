import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary] Uncaught render error:', error, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center gap-4 p-8 min-h-[300px] text-center">
          <AlertCircle className="h-10 w-10 text-destructive opacity-80" aria-hidden="true" />
          <div className="space-y-1">
            <p className="text-sm font-semibold text-foreground">Something went wrong</p>
            {this.state.error && (
              <p className="text-xs text-muted-foreground font-mono max-w-md break-words">
                {this.state.error.message}
              </p>
            )}
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => window.location.reload()}
          >
            Reload page
          </Button>
        </div>
      )
    }

    return this.props.children
  }
}
