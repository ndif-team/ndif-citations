import { useNavigate } from 'react-router-dom'
import {
  CheckCircle2,
  Clock,
  XCircle,
  GitBranch,
  Play,
  Upload,
  AlertCircle,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useStats } from '@/api/hooks'
import { CategoryChart } from '@/components/dashboard/CategoryChart'

function KpiCard({
  label,
  value,
  sub,
  icon: Icon,
  accentClass,
}: {
  label: string
  value: number | string
  sub?: string
  icon: React.ElementType
  accentClass: string
}) {
  return (
    <Card>
      <CardHeader className="pb-1">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            {label}
          </CardTitle>
          <div className={`p-1.5 rounded-md ${accentClass}`}>
            <Icon className="h-3.5 w-3.5" aria-hidden="true" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold tabular-nums">{value}</div>
        {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
      </CardContent>
    </Card>
  )
}

function KpiSkeleton() {
  return (
    <Card>
      <CardHeader className="pb-1">
        <Skeleton className="h-3 w-24" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-8 w-16 mt-1" />
        <Skeleton className="h-3 w-32 mt-2" />
      </CardContent>
    </Card>
  )
}

export function Dashboard() {
  const { data: stats, isLoading, error } = useStats()
  const navigate = useNavigate()

  if (error) {
    return (
      <div className="flex items-center gap-2 text-destructive text-sm p-4">
        <AlertCircle className="h-4 w-4 flex-none" aria-hidden="true" />
        <span>Failed to load stats: {(error as Error).message}</span>
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold">Dashboard</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Curation pipeline overview
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            onClick={() => navigate('/runs')}
            className="gap-1.5"
          >
            <Play className="h-3.5 w-3.5" />
            Start a run
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => navigate('/publish')}
            className="gap-1.5"
          >
            <Upload className="h-3.5 w-3.5" />
            Publish to site
          </Button>
        </div>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {isLoading ? (
          <>
            <KpiSkeleton />
            <KpiSkeleton />
            <KpiSkeleton />
            <KpiSkeleton />
          </>
        ) : stats ? (
          <>
            <KpiCard
              label="Verified"
              value={stats.papers.verified}
              sub={`of ${stats.papers.total} total`}
              icon={CheckCircle2}
              accentClass="bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-400"
            />
            <KpiCard
              label="Pending"
              value={stats.papers.pending}
              sub="awaiting review"
              icon={Clock}
              accentClass="bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-400"
            />
            <KpiCard
              label="Discarded"
              value={stats.papers.discarded}
              sub="not relevant"
              icon={XCircle}
              accentClass="bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
            />
            <KpiCard
              label="Repos"
              value={stats.repos.total}
              sub={`${stats.repos.research}R · ${stats.repos.course}C · ${stats.repos.experiment}E`}
              icon={GitBranch}
              accentClass="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-400"
            />
          </>
        ) : null}
      </div>

      {/* Category chart */}
      <div className="grid md:grid-cols-2 gap-4">
        {isLoading ? (
          <Card>
            <CardHeader>
              <Skeleton className="h-4 w-32" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-48 w-full" />
            </CardContent>
          </Card>
        ) : stats ? (
          <CategoryChart categories={stats.categories} />
        ) : null}

        {/* Quick stats card */}
        {stats && (
          <Card>
            <CardHeader>
              <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Breakdown
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1.5">Papers by bucket</p>
                  <div className="space-y-1.5">
                    {[
                      { label: 'Verified', value: stats.papers.verified, total: stats.papers.total, color: 'bg-green-500' },
                      { label: 'Pending', value: stats.papers.pending, total: stats.papers.total, color: 'bg-amber-500' },
                      { label: 'Discarded', value: stats.papers.discarded, total: stats.papers.total, color: 'bg-slate-400' },
                    ].map(({ label, value, total, color }) => (
                      <div key={label} className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full flex-none ${color}`} aria-hidden="true" />
                        <span className="text-xs text-muted-foreground w-16">{label}</span>
                        <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${color}`}
                            style={{ width: total > 0 ? `${(value / total) * 100}%` : '0%' }}
                          />
                        </div>
                        <span className="text-xs tabular-nums w-8 text-right">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1.5">Repos by type</p>
                  <div className="flex gap-4">
                    {[
                      { label: 'Research', value: stats.repos.research },
                      { label: 'Course', value: stats.repos.course },
                      { label: 'Experiment', value: stats.repos.experiment },
                    ].map(({ label, value }) => (
                      <div key={label} className="text-center">
                        <p className="text-lg font-bold tabular-nums">{value}</p>
                        <p className="text-xs text-muted-foreground">{label}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
