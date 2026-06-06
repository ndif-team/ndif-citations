/**
 * Mirrors edit_schema.EDITABLE_FIELDS (16 fields) for the frontend edit form.
 * Each entry carries a label, an input type, and optional options for select
 * inputs.
 */

export type EditInputType =
  | 'text'
  | 'number'
  | 'textarea'
  | 'select'

export interface SelectOption {
  value: string
  label: string
}

export interface EditableFieldMeta {
  name: string
  label: string
  type: EditInputType
  /** Only set when type === 'select' */
  options?: SelectOption[]
  /** Placeholder text */
  placeholder?: string
}

/**
 * Sentinel value used in Select components to represent the "none / unset /
 * clear" option.  Radix UI's Select forbids `value=""` (empty string is
 * reserved for its internal clear mechanism), so we use this non-empty
 * constant instead.  All code that reads from / writes to the edit form must
 * map between the sentinel and the real cleared representation:
 *
 *   UI value   → API payload
 *   SELECT_NONE → ""   (which the backend parses as None for reason /
 *                        peer_reviewed)
 */
export const SELECT_NONE = '__none__'

const CATEGORY_OPTIONS: SelectOption[] = [
  { value: 'uses_ndif',      label: 'Uses NDIF' },
  { value: 'uses_nnsight',   label: 'Uses NNsight' },
  { value: 'referencing',    label: 'Referencing' },
  { value: 'unclassified',   label: 'Unclassified' },
]

const BUCKET_OPTIONS: SelectOption[] = [
  { value: 'pending',    label: 'Pending' },
  { value: 'verified',   label: 'Verified' },
  { value: 'discarded',  label: 'Discarded' },
]

/**
 * PaperReason values from ndif_citations.models.PaperReason — must stay in
 * sync with the backend enum.
 *
 * The "none" option uses SELECT_NONE instead of "" because Radix UI Select
 * forbids value="".
 */
const REASON_OPTIONS: SelectOption[] = [
  { value: SELECT_NONE,                label: '(none)' },
  { value: 'openalex_source',          label: 'OpenAlex source' },
  { value: 'low_confidence',           label: 'Low confidence' },
  { value: 'medium_confidence',        label: 'Medium confidence' },
  { value: 'unclassified_no_keywords', label: 'Unclassified — no keywords' },
  { value: 'unclassified_llm',         label: 'Unclassified — LLM' },
  { value: 'stub_metadata',            label: 'Stub metadata' },
  { value: 'zero_pdf_hits',            label: 'Zero PDF hits' },
  { value: 'manual_discard',           label: 'Manual discard' },
  { value: 'manual_demote',            label: 'Manual demote' },
]

const PEER_REVIEWED_OPTIONS: SelectOption[] = [
  { value: SELECT_NONE, label: '(unset)' },
  { value: 'yes',       label: 'Yes' },
  { value: 'no',        label: 'No' },
]

/** All 16 editable fields, in schema order. */
export const EDITABLE_FIELDS: EditableFieldMeta[] = [
  { name: 'title',        label: 'Title',        type: 'text',     placeholder: 'Paper title' },
  { name: 'authors',      label: 'Authors',      type: 'text',     placeholder: 'Comma-separated author list' },
  { name: 'affiliations', label: 'Affiliations', type: 'text',     placeholder: 'Comma-separated institutions' },
  { name: 'venue',        label: 'Venue',        type: 'text',     placeholder: 'Conference / journal / "ArXiv YYYY"' },
  { name: 'year',         label: 'Year',         type: 'number',   placeholder: 'Publication year' },
  { name: 'category',     label: 'Category',     type: 'select',   options: CATEGORY_OPTIONS },
  { name: 'description',  label: 'Description',  type: 'textarea', placeholder: '1–3 sentence website summary' },
  { name: 'url',          label: 'URL',          type: 'text',     placeholder: 'Landing page URL' },
  { name: 'pdf_url',      label: 'PDF URL',      type: 'text',     placeholder: 'Direct PDF link (empty to clear)' },
  { name: 'project_url',  label: 'Project URL',  type: 'text',     placeholder: 'GitHub or project page (empty to clear)' },
  { name: 'image',        label: 'Image path',   type: 'text',     placeholder: '/images/Slug.png (empty to clear)' },
  { name: 'bucket',       label: 'Bucket',       type: 'select',   options: BUCKET_OPTIONS },
  { name: 'reason',       label: 'Reason',       type: 'select',   options: REASON_OPTIONS },
  { name: 'reason_detail',label: 'Reason detail',type: 'text',     placeholder: 'Free-text supplement' },
  { name: 'peer_reviewed',label: 'Peer reviewed',type: 'select',   options: PEER_REVIEWED_OPTIONS },
  { name: 'abstract',     label: 'Abstract',     type: 'textarea', placeholder: 'Full abstract text' },
]

/** Lookup a field meta by name. */
export function getEditableField(name: string): EditableFieldMeta | undefined {
  return EDITABLE_FIELDS.find(f => f.name === name)
}
