import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { StatsResponse } from '@/api/types'

const CATEGORY_CONFIG: Record<
  string,
  { label: string; color: string; darkColor: string }
> = {
  uses_ndif: { label: 'Uses NDIF', color: '#1E40AF', darkColor: '#60A5FA' },
  uses_nnsight: { label: 'Uses NNsight', color: '#7C3AED', darkColor: '#A78BFA' },
  referencing: { label: 'Referencing', color: '#0E7490', darkColor: '#22D3EE' },
  unclassified: { label: 'Unclassified', color: '#64748B', darkColor: '#94A3B8' },
}

interface Props {
  categories: StatsResponse['categories']
}

export function CategoryChart({ categories }: Props) {
  const isDark = document.documentElement.classList.contains('dark')

  const data = Object.entries(categories).map(([key, value]) => ({
    key,
    label: CATEGORY_CONFIG[key]?.label ?? key,
    value,
    color: isDark
      ? (CATEGORY_CONFIG[key]?.darkColor ?? '#94A3B8')
      : (CATEGORY_CONFIG[key]?.color ?? '#64748B'),
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Category distribution
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div aria-label="Category distribution chart" role="img">
          <ResponsiveContainer width="100%" height={160}>
            <BarChart
              data={data}
              margin={{ top: 4, right: 8, left: -16, bottom: 0 }}
              barCategoryGap="30%"
            >
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fill: 'hsl(215 20% 45%)' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 10, fill: 'hsl(215 20% 45%)' }}
                axisLine={false}
                tickLine={false}
                allowDecimals={false}
              />
              <Tooltip
                contentStyle={{
                  fontSize: 12,
                  borderRadius: 6,
                  border: '1px solid hsl(214 32% 88%)',
                  background: 'hsl(0 0% 100%)',
                }}
                cursor={{ fill: 'hsl(210 40% 94%)' }}
                formatter={(value) => [value, '']}
                labelFormatter={(_label, payload) =>
                  payload && payload[0] ? String(payload[0].payload.label) : ''
                }
              />
              <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                {data.map((entry) => (
                  <Cell key={entry.key} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        {/* Legend — color not the only signal */}
        <div className="flex flex-wrap gap-3 mt-3">
          {data.map(({ key, label, color }) => (
            <div key={key} className="flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-sm flex-none"
                style={{ background: color }}
                aria-hidden="true"
              />
              <span className="text-xs text-muted-foreground">{label}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
