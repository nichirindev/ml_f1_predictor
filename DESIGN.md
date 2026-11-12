# Pitwall design system

A scenario analysis dashboard for F1 fans and ML students. Primary task: compare
podium estimates under explicit assumptions. Precise, sporting, restrained,
transparent. Aesthetic essence: motorsport timing desk.

Considered a broadcast timing desk, a white engineering notebook, and a magazine
race report. The timing desk best fits dense comparisons and grid editing.
Its signature is the ranked timing sheet: starting slot, driver, team, probability,
then model disagreement. No decorative imagery or artificial live indicators.

## Tokens

Plain CSS custom properties; Streamlit theme supplies native component states.
Body: Barlow; display: Barlow Condensed; local sans-serif fallback for offline use.
Base 16px, ratio 1.2: 14, 16, 19, 23, 28, 34px. Spacing unit 4px.
Radii 4px and 8px. Defined edges, no shadows.

| Role | OKLCH |
|---|---|
| Background | oklch(19.5% .014 260) |
| Surface | oklch(25% .018 260) |
| Foreground | oklch(95% .006 260) |
| Muted | oklch(77% .018 260) |
| Border | oklch(39% .02 260) |
| Accent | oklch(69% .21 28) |
| Accent foreground | oklch(19.5% .014 260) |
| Success | oklch(79% .13 160) |
| Warning | oklch(84% .13 85) |
| Error | oklch(76% .16 28) |

## Craft

Sidebar houses shared assumptions. Work area starts with a compact title and an
explicit scenario notice; analysis uses native tabs, paired comparisons, and a
sortable timing sheet. Numbers use tabular figures; headings use condensed type.
Primary emphasis goes to the estimate, not decoration. Tables retain native
keyboard behavior. Grid validation preserves edits and blocks invalid scoring.
Score charts use a fixed 0–100% axis and percentage tooltips. Controls have visible labels and native loading, focus, selected and disabled
states. No motion, custom icons, or color-only meaning. Layout stacks on mobile.
Historical team mappings are disclosed; scores are never called calibrated
confidence or race-finish probabilities. Model agreement is not uncertainty.

## Audit

2026-11-13: Source audit passes palette, hierarchy, restrained geometry, labeled
controls, non-color status, and reduced-motion requirements. Desktop and 390px mobile browser checks pass, with no JavaScript errors or page
overflow. CSV export and all four tabs were exercised; five automated tests pass. No accessibility
conformance claim without a full assistive-technology audit.
