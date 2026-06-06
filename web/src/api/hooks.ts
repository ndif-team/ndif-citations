import { useQuery } from '@tanstack/react-query'
import { fetchStats, fetchPapers, fetchPaper, fetchRuns, fetchRun, fetchActiveRun, fetchRepos, fetchRepo } from './client'
import type { PapersParams, ReposParams } from './client'

export function useStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    staleTime: 30_000,
    refetchInterval: 60_000,
  })
}

export function usePapers(params: PapersParams = {}) {
  return useQuery({
    queryKey: ['papers', params],
    queryFn: () => fetchPapers(params),
    staleTime: 10_000,
  })
}

export function usePaper(id: string | null) {
  return useQuery({
    queryKey: ['paper', id],
    queryFn: () => fetchPaper(id!),
    enabled: !!id,
    staleTime: 60_000,
  })
}

export function useRuns() {
  return useQuery({
    queryKey: ['runs'],
    queryFn: fetchRuns,
    staleTime: 10_000,
    refetchInterval: 15_000,
  })
}

export function useRun(runId: string | null) {
  return useQuery({
    queryKey: ['run', runId],
    queryFn: () => fetchRun(runId!),
    enabled: !!runId,
    staleTime: 5_000,
  })
}

export function useActiveRun() {
  return useQuery({
    queryKey: ['runs', 'active'],
    queryFn: fetchActiveRun,
    staleTime: 5_000,
    refetchInterval: 10_000,
  })
}

export function useRepos(params: ReposParams = {}) {
  return useQuery({
    queryKey: ['repos', params],
    queryFn: () => fetchRepos(params),
    staleTime: 30_000,
  })
}

export function useRepo(owner: string | null, repo: string | null) {
  return useQuery({
    queryKey: ['repo', owner, repo],
    queryFn: () => fetchRepo(owner!, repo!),
    enabled: !!(owner && repo),
    staleTime: 60_000,
  })
}
