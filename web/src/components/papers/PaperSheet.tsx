import { ExternalLink, Copy, Check, AlertCircle } from 'lucide-react'
import { useState } from 'react'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { usePaper } from '@/api/hooks'
import { bucketBadge, confidenceBadge, categoryBadge, categoryLabel } from '@/lib/tokens'
import type { Bucket, ConfidenceBand, Category } from '@/api/types'

interface Props {
  paperId: string | null
  onClose: () => void
}

function ExternalLinkButton({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
    >
      <ExternalLink className="h-3 w-3" aria-hidden="true" />
      {label}
    </a>
  )
}

function CopyBibtex({ bibtex }: { bibtex: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(bibtex)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">BibTeX</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          className="h-6 px-2 text-xs gap-1"
          aria-label={copied ? 'Copied' : 'Copy BibTeX'}
        >
          {copied ? (
            <><Check className="h-3 w-3 text-green-500" />Copied</>
          ) : (
            <><Copy className="h-3 w-3" />Copy</>
          )}
        </Button>
      </div>
      <pre className="text-[10px] leading-relaxed font-mono bg-muted p-2.5 rounded-md overflow-x-auto text-muted-foreground whitespace-pre-wrap break-all">
        {bibtex}
      </pre>
    </div>
  )
}

export function PaperSheet({ paperId, onClose }: Props) {
  const { data: paper, isLoading, error } = usePaper(paperId)

  const isOpen = !!paperId

  return (
    <Sheet open={isOpen} onOpenChange={(open) => { if (!open) onClose() }}>
      <SheetContent side="right" className="overflow-y-auto p-0 flex flex-col">
        {isLoading && (
          <div className="p-6 space-y-4">
            <Skeleton className="h-5 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
            <Skeleton className="h-40 w-full rounded-md" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-4/6" />
          </div>
        )}

        {error && (
          <div className="p-6 flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="h-4 w-4 flex-none" aria-hidden="true" />
            <span>Failed to load paper: {(error as Error).message}</span>
          </div>
        )}

        {paper && (
          <>
            {/* Thumbnail */}
            {paper.has_image && paper.image && (
              <div className="p-4 bg-muted/30 border-b">
                <img
                  src={`/api/images/${encodeURIComponent(paper.image.replace(/^\/images\//, ''))}`}
                  alt={`Thumbnail for ${paper.title}`}
                  className="w-full max-h-48 object-contain rounded-md"
                  loading="lazy"
                />
              </div>
            )}

            <SheetHeader className="px-5 pt-5 pb-3">
              <SheetTitle className="text-sm font-semibold leading-snug pr-6">
                {paper.title}
              </SheetTitle>
              {paper.authors && paper.authors.length > 0 && (
                <SheetDescription className="text-xs">
                  {paper.authors.join(', ')}
                </SheetDescription>
              )}
            </SheetHeader>

            <div className="px-5 pb-5 space-y-4 flex-1">
              {/* Badges row */}
              <div className="flex flex-wrap gap-1.5">
                <span className={bucketBadge(paper.bucket as Bucket)}>
                  {paper.bucket}
                </span>
                <span className={categoryBadge(paper.category as Category)}>
                  {categoryLabel(paper.category as Category)}
                </span>
                <span className={confidenceBadge(paper.confidence_band as ConfidenceBand)}>
                  {paper.confidence_band}
                </span>
              </div>

              {/* Venue & year */}
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                {paper.venue && (
                  <span>
                    <span className="font-medium text-foreground">{paper.venue}</span>
                    {paper.venue_source && (
                      <span className="ml-1 opacity-60">({paper.venue_source})</span>
                    )}
                  </span>
                )}
                {paper.year && (
                  <span className="tabular-nums">{paper.year}</span>
                )}
              </div>

              {/* Affiliations */}
              {paper.affiliations && paper.affiliations.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Affiliations</p>
                  <p className="text-xs text-muted-foreground">{paper.affiliations.join('; ')}</p>
                </div>
              )}

              {/* Reason */}
              {paper.reason && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Classification reason</p>
                  <p className="text-xs text-foreground">{paper.reason}</p>
                  {paper.reason_detail && (
                    <p className="text-xs text-muted-foreground mt-0.5">{paper.reason_detail}</p>
                  )}
                </div>
              )}

              {/* Abstract */}
              {paper.abstract && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Abstract</p>
                  <p className="text-xs leading-relaxed text-foreground line-clamp-[12]">{paper.abstract}</p>
                </div>
              )}

              {/* Links */}
              {(paper.url || paper.pdf_url || paper.project_url) && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Links</p>
                  <div className="flex flex-wrap gap-3">
                    {paper.url && <ExternalLinkButton href={paper.url} label="Paper" />}
                    {paper.pdf_url && <ExternalLinkButton href={paper.pdf_url} label="PDF" />}
                    {paper.project_url && <ExternalLinkButton href={paper.project_url} label="Project" />}
                  </div>
                </div>
              )}

              {/* ID */}
              <div>
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">ID</p>
                <code className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                  {paper.id}
                </code>
              </div>

              {/* BibTeX */}
              {paper.bibtex && <CopyBibtex bibtex={paper.bibtex} />}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
