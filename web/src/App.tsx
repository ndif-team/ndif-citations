import { lazy } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import { AppShell } from '@/components/layout/AppShell'
import { ErrorBoundary } from '@/components/ErrorBoundary'

// Route-level code splitting: each page becomes its own chunk so heavy, page-local
// deps (e.g. recharts on the Dashboard) stay out of the initial bundle. The Suspense
// boundary lives in AppShell so the sidebar/topbar persist while a chunk loads.
const Dashboard = lazy(() => import('@/pages/Dashboard').then((m) => ({ default: m.Dashboard })))
const Papers = lazy(() => import('@/pages/Papers').then((m) => ({ default: m.Papers })))
const Repos = lazy(() => import('@/pages/Repos').then((m) => ({ default: m.Repos })))
const Runs = lazy(() => import('@/pages/Runs').then((m) => ({ default: m.Runs })))
const Publish = lazy(() => import('@/pages/Publish').then((m) => ({ default: m.Publish })))
const Settings = lazy(() => import('@/pages/Settings').then((m) => ({ default: m.Settings })))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
            <Route path="/papers" element={<ErrorBoundary><Papers /></ErrorBoundary>} />
            <Route path="/repos" element={<ErrorBoundary><Repos /></ErrorBoundary>} />
            <Route path="/runs" element={<ErrorBoundary><Runs /></ErrorBoundary>} />
            <Route path="/publish" element={<ErrorBoundary><Publish /></ErrorBoundary>} />
            <Route path="/settings" element={<ErrorBoundary><Settings /></ErrorBoundary>} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster richColors position="bottom-right" />
    </QueryClientProvider>
  )
}

export default App
