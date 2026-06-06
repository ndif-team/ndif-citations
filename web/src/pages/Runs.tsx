import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Play } from 'lucide-react'

export function Runs() {
  return (
    <div className="max-w-lg">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Play className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
            Runs
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Pipeline runs browser coming soon.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
