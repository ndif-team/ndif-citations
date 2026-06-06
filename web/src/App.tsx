import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import { AppShell } from '@/components/layout/AppShell'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { Dashboard } from '@/pages/Dashboard'
import { Papers } from '@/pages/Papers'
import { Repos } from '@/pages/Repos'
import { Runs } from '@/pages/Runs'
import { Settings } from '@/pages/Settings'

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
            <Route path="/settings" element={<ErrorBoundary><Settings /></ErrorBoundary>} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster richColors position="bottom-right" />
    </QueryClientProvider>
  )
}

export default App
