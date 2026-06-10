import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  FileText,
  GitBranch,
  Play,
  UploadCloud,
  Settings,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/papers', label: 'Papers', icon: FileText, end: false },
  { to: '/repos', label: 'Repos', icon: GitBranch, end: false },
  { to: '/runs', label: 'Runs', icon: Play, end: false },
  { to: '/publish', label: 'Publish', icon: UploadCloud, end: false },
  { to: '/settings', label: 'Settings', icon: Settings, end: false },
]

export function Sidebar() {
  return (
    <nav
      className="sidebar flex-none flex flex-col w-52 h-full"
      aria-label="Main navigation"
    >
      {/* Logo area */}
      <div className="flex items-center gap-2.5 px-4 h-12 border-b border-[hsl(var(--sidebar-border))]">
        <div className="w-6 h-6 rounded bg-[hsl(var(--sidebar-active))] flex items-center justify-center flex-none">
          <span className="text-[9px] font-bold text-[hsl(222_47%_8%)] leading-none">ND</span>
        </div>
        <span className="text-xs font-semibold text-[hsl(var(--sidebar-fg))] truncate">Citations</span>
      </div>

      {/* Nav links */}
      <ul className="flex flex-col gap-0.5 p-2 flex-1" role="list">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  'sidebar-nav-item flex items-center gap-2.5 px-3 py-2 text-sm w-full',
                  isActive && 'active'
                )
              }
            >
              <Icon className="h-4 w-4 flex-none" aria-hidden="true" />
              {label}
            </NavLink>
          </li>
        ))}
      </ul>

      {/* Footer */}
      <div className="p-3 border-t border-[hsl(var(--sidebar-border))]">
        <p className="text-[10px] text-[hsl(var(--sidebar-fg)/0.4)] font-mono">v2.4.0</p>
      </div>
    </nav>
  )
}
