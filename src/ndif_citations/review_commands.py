"""Slash commands in review / digest / report / intake issues (spec §6.3)."""
from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

from ndif_citations.digest import _SAFE_URL
from ndif_citations.models import Category
from ndif_citations.output import load_existing_papers
from ndif_citations.review_render import BOT_LOGINS, HELP_TEXT, parse_marker
from ndif_citations.review_sync import resolve_paper
from ndif_citations.utils import extract_arxiv_id_from_url

CATEGORIES = {"uses_ndif", "uses_nnsight", "referencing"}
EDIT_FIELDS = {"title": "title", "authors": "authors", "affiliations": "affiliations", "venue": "venue",
               "year": "year", "description": "description", "summary": "description",
               "url": "url", "project_url": "project_url", "github": "project_url"}
_DOI_URL = re.compile(r"doi\.org/(10\.\S+)", re.I)
_TARGETED = {"digest", "report", "intake"}  # issues whose commands must name the paper explicitly


@dataclass
class Command:
    name: str
    args: str


@dataclass
class ReviewContext:
    out: Path
    issue_number: int
    issue_body: str
    issue_author: str
    issue_labels: set[str]
    comment_body: str
    comment_author: str


@dataclass
class Result:
    replies: list[str] = field(default_factory=list)
    changed: bool = False
    close: bool = False

    @property
    def reply(self) -> str:
        return "\n\n".join(self.replies)


class UserError(Exception):
    """A problem the commenter can fix — replied, not treated as a failure."""


def parse_commands(text: str, limit: int = 10) -> list[Command]:
    cmds: list[Command] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("/") and len(s) > 1 and not s.startswith("//"):
            name, _, rest = s[1:].partition(" ")
            cmds.append(Command(name.lower(), rest.strip()))
    return cmds[:limit]


def _ident_from_token(token: str) -> dict | None:
    t = token.strip()
    if t.lower().startswith("arxiv:"):
        return {"arxiv": t[6:]}
    if t.lower().startswith("doi:"):
        return {"doi": t[4:]}
    if t.startswith(("http://", "https://")):
        m = _DOI_URL.search(t)
        return {"url": t, "arxiv": extract_arxiv_id_from_url(t) or "", "doi": m.group(1) if m else ""}
    return None


def _bot_review_issue(ctx: ReviewContext) -> bool:
    """A bot-opened per-paper review issue: its marker is trusted."""
    return "review" in ctx.issue_labels and ctx.issue_author in BOT_LOGINS and not ctx.issue_labels & _TARGETED


def _target(ctx: ReviewContext, args: str) -> tuple[str, str]:
    """(paper merge_key, remaining args). Raises UserError when no trusted target exists."""
    papers = load_existing_papers(ctx.out)
    if ctx.issue_labels & _TARGETED:
        token, _, rest = args.partition(" ")
        ident = _ident_from_token(token)
        if ident is None:
            raise UserError("Put the paper's URL (or `arxiv:<id>` / `doi:<doi>`) right after the command.")
        paper = resolve_paper(papers, ident)
        if paper is None:
            raise UserError(f"No paper in the catalog matches `{token}`.")
        return paper.merge_key(), rest.strip()
    if _bot_review_issue(ctx):
        paper = resolve_paper(papers, parse_marker(ctx.issue_body))
        if paper is None:
            raise UserError("I can't find this issue's paper in the catalog anymore.")
        return paper.merge_key(), args
    raise UserError("Commands work in bot-opened review issues, or with an explicit paper URL in digest/report issues.")


def _paper(ctx: ReviewContext, key: str):
    return next(p for p in load_existing_papers(ctx.out) if p.merge_key() == key)


def _state(ctx: ReviewContext, key: str) -> str:
    p = _paper(ctx, key)
    conf = "curator-locked" if p.manual_override else f"confidence {p.category_confidence_band.value}"
    return f"now **{p.bucket.value}** · {p.category.value} ({conf})"


def cmd_verify(ctx: ReviewContext, key: str, args: str, res: Result) -> str:
    from ndif_citations.server.services import papers_svc
    cat = args.strip().lower() or None
    if cat and cat not in CATEGORIES:
        raise UserError(f"Unknown category `{cat}` — use one of: {', '.join(sorted(CATEGORIES))}.")
    paper = _paper(ctx, key)
    if cat is None and paper.category == Category.UNCLASSIFIED:
        raise UserError("This paper needs a category: `/verify uses_nnsight`, `/verify uses_ndif`, or `/verify referencing`.")
    if cat:
        papers_svc.edit_paper(ctx.out, key, {"category": cat, "bucket": paper.bucket.value})
    papers_svc.set_bucket(ctx.out, key, "verified", None, f"verified via #{ctx.issue_number} by @{ctx.comment_author}")
    res.changed = True
    res.close |= _bot_review_issue(ctx)  # the paper is resolved; digest/report issues stay open
    return f"Verified — {_state(ctx, key)}. ndif.us updates within a day."


def cmd_discard(ctx: ReviewContext, key: str, args: str, res: Result) -> str:
    from ndif_citations.server.services import papers_svc
    detail = args.strip() or f"discarded via #{ctx.issue_number} by @{ctx.comment_author}"
    papers_svc.set_bucket(ctx.out, key, "discarded", "manual_discard", detail)
    res.changed = True
    res.close |= _bot_review_issue(ctx)
    return "Discarded."


def cmd_edit(ctx: ReviewContext, key: str, args: str, res: Result) -> str:
    from ndif_citations.server.services import papers_svc
    name, sep, value = args.partition(":")
    field_name = EDIT_FIELDS.get(name.strip().lower())
    if not sep or field_name is None:
        raise UserError(f"I can't edit `{name.strip()}` — use one of: {', '.join(sorted(EDIT_FIELDS))} "
                        "(as `/edit field: value`).")
    value = value.strip()
    if not value:
        raise UserError(f"Give {field_name} a value (as `/edit {name.strip()}: value`).")
    if field_name in ("url", "project_url") and not _SAFE_URL.fullmatch(value):
        raise UserError(f"Couldn't set {field_name}: use an http(s) link.")
    try:
        # No explicit bucket: edit_paper's _bucket_after_edit never publishes, keeps verified/discarded,
        # and refreshes a pending reason (e.g. /edit year clears stub_metadata → why: label updates).
        papers_svc.edit_paper(ctx.out, key, {field_name: value})
    except ValueError:
        raise UserError(f"Couldn't set {field_name}: check the value's format (e.g. `/edit year: 2025`).")
    res.changed = True
    return f"Updated {field_name}."


def cmd_evidence(ctx: ReviewContext, key: str, args: str, res: Result) -> str:
    from ndif_citations.server.services import papers_svc
    try:
        papers_svc.add_curator_quote(ctx.out, key, args.strip().strip('"“”'))
    except ValueError:  # empty / over 2,000 chars; never echo the exception text
        raise UserError("Paste the sentence right after `/evidence` (up to 2,000 characters).")
    res.changed = True
    return "Quote saved as evidence. If it shows the paper used NDIF/NNsight, follow up with `/verify <category>`."


Handler = Callable[[ReviewContext, str, str, Result], str]
HANDLERS: dict[str, Handler] = {
    "verify": cmd_verify, "discard": cmd_discard, "edit": cmd_edit, "evidence": cmd_evidence,
}


def run_commands(ctx: ReviewContext) -> Result:
    res = Result()
    cmds = parse_commands(ctx.comment_body)
    if not cmds:
        res.replies.append(HELP_TEXT)
        return res
    for cmd in cmds:
        if cmd.name == "help":
            res.replies.append(HELP_TEXT)
            continue
        handler = HANDLERS.get(cmd.name)
        if handler is None:
            res.replies.append(f"⚠️ Unknown command `/{cmd.name}`.\n\n{HELP_TEXT}")
            continue
        try:
            key, rest = _target(ctx, cmd.args) if cmd.name != "add" else ("", cmd.args)
            res.replies.append(f"✅ `/{cmd.name}` — {handler(ctx, key, rest, res)}")
        except UserError as e:
            res.replies.append(f"⚠️ `/{cmd.name}` — {e}")
    return res

_MAX_PDF = 50 * 1024 * 1024
_MAX_IMG = 10 * 1024 * 1024
_IMG_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)|<img[^>]+src=\"(https?://[^\"]+)\"", re.I)
_DOWNLOAD_DEADLINE = 120  # seconds, total per download
_ATTACHMENT_PATH = re.compile(r"/user-attachments/assets/[A-Za-z0-9-]+")
_IMG_HOSTS = {"private-user-images.githubusercontent.com", "user-images.githubusercontent.com"}


def _download(url: str, max_bytes: int) -> bytes:
    """https-only bounded download (redirects allowed — GitHub attachments 302 to signed URLs)."""
    if urlparse(url).scheme != "https":
        raise UserError("Only https:// links are accepted.")
    resp = None
    try:
        start = time.monotonic()
        resp = requests.get(url, stream=True, timeout=60, allow_redirects=True,
                            headers={"User-Agent": "NDIFCitationTracker/1.0 (https://ndif.us)"})
        resp.raise_for_status()
        hops = [*getattr(resp, "history", []), resp]
        if any(urlparse(getattr(h, "url", url)).scheme != "https" for h in hops):
            raise UserError("That link redirects to a non-https address.")
        chunks, total = [], 0
        for chunk in resp.iter_content(chunk_size=65536):
            if time.monotonic() - start > _DOWNLOAD_DEADLINE:
                raise UserError("The download took too long.")
            total += len(chunk)
            if total > max_bytes:
                raise UserError(f"File is larger than {max_bytes // (1024 * 1024)} MB.")
            chunks.append(chunk)
        return b"".join(chunks)
    except ValueError:  # InvalidURL, bad port, http://[::1 ... (first: InvalidURL is also a RequestException)
        raise UserError("That isn't a valid link.")
    except requests.RequestException:
        raise UserError("Couldn't download that link (it may need a login or have expired).")
    finally:
        if resp is not None:
            resp.close()


def _llm_ready() -> None:
    from ndif_citations.preflight import preflight
    pf = preflight(skip_papers=False, skip_github=True, validate_llm=True)
    if not pf["ok"]:
        raise UserError("The LLM is unavailable right now, so I didn't change anything: "
                        + "; ".join(pf["blocking"]) + " — try again later.")


def cmd_pdf(ctx: ReviewContext, key: str, args: str, res: Result) -> str:
    from ndif_citations.reprocess import reprocess_papers
    from ndif_citations.server.services import papers_svc
    url = args.strip().split()[0] if args.strip() else ""
    if not url:
        raise UserError("Give an open-access PDF link: `/pdf https://…`.")
    if urlparse(url).scheme != "https":
        raise UserError("Only https:// links are accepted.")
    _llm_ready()
    data = _download(url, _MAX_PDF)
    if not data.startswith(b"%PDF-"):
        raise UserError("That link isn't a PDF (it may be a landing page — link the PDF itself).")
    papers_svc.attach_pdf(ctx.out, key, data)
    papers_svc.edit_paper(ctx.out, key, {"pdf_url": url, "bucket": _paper(ctx, key).bucket.value})
    papers_svc.backfill_evidence(ctx.out, key)
    # Never replace an existing figure (it may be a curator's /thumbnail screenshot).
    reprocess_papers(ctx.out, [key], ["classify"] if _paper(ctx, key).image else ["classify", "thumbnail"])
    res.changed = True
    return f"PDF attached, evidence re-extracted, re-classified — {_state(ctx, key)}."


def cmd_thumbnail(ctx: ReviewContext, key: str, args: str, res: Result) -> str:
    import io

    from PIL import Image
    from ndif_citations.pdf_cache import get_cached_pdf
    from ndif_citations.reprocess import reprocess_papers
    from ndif_citations.server.services import papers_svc
    m = _IMG_RE.search(ctx.comment_body)
    if m:
        url = m.group(1) or m.group(2)
        u = urlparse(url)
        ok = u.scheme == "https" and (u.netloc in _IMG_HOSTS or
                                      (u.netloc == "github.com" and _ATTACHMENT_PATH.fullmatch(u.path)))
        if not ok:
            raise UserError("Paste the screenshot directly into the comment (it must be a GitHub attachment).")
        data = _download(url, _MAX_IMG)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                img = Image.open(io.BytesIO(data), formats=["PNG", "JPEG", "GIF", "WEBP"])
                img.load()
        except Exception:  # noqa: BLE001 — any decode failure is the user's file
            raise UserError("That attachment isn't an image I can read.")
        img.thumbnail((1600, 1600))
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            flat = Image.new("RGB", rgba.size, "white")
            flat.paste(rgba, mask=rgba.split()[-1])
            img = flat
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        papers_svc.save_image_bytes(ctx.out, key, buf.getvalue())
        res.changed = True
        return "Thumbnail set."
    if get_cached_pdf(_paper(ctx, key), ctx.out) is None:
        raise UserError("no PDF available to extract a figure from — use `/pdf <open-access link>` "
                        "or paste a screenshot with `/thumbnail`.")
    def _img_bytes():
        img = _paper(ctx, key).image
        f = ctx.out / img.lstrip("/") if img else None
        return f.read_bytes() if f is not None and f.exists() else None

    before = _img_bytes()
    reprocess_papers(ctx.out, [key], ["thumbnail"])
    res.changed = True  # reprocess rewrote the catalog (lock) either way
    after = _img_bytes()
    if after is None or after == before:
        return ("No new figure found in the PDF; kept the existing thumbnail." if before is not None
                else "No new figure found.")
    return "Thumbnail re-extracted from the PDF."


def cmd_retry(ctx: ReviewContext, key: str, args: str, res: Result) -> str:
    from ndif_citations.pdf_cache import get_cached_pdf
    from ndif_citations.reprocess import reprocess_papers
    _llm_ready()
    if get_cached_pdf(_paper(ctx, key), ctx.out) is None:
        raise UserError("no PDF available (none found online) — use `/pdf <open-access link>`, or for a "
                        "paywalled paper paste the key sentence with `/evidence` and then `/verify`.")
    reprocess_papers(ctx.out, [key], ["classify"] if _paper(ctx, key).image else ["classify", "thumbnail"])
    res.changed = True
    return f"Re-classified — {_state(ctx, key)}."


def cmd_summarize(ctx: ReviewContext, key: str, args: str, res: Result) -> str:
    from ndif_citations.reprocess import reprocess_papers
    _llm_ready()
    reprocess_papers(ctx.out, [key], ["summary"])
    res.changed = True
    return "Summary regenerated."


HANDLERS.update({"pdf": cmd_pdf, "thumbnail": cmd_thumbnail, "retry": cmd_retry, "summarize": cmd_summarize})
