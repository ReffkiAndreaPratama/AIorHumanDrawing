/**
 * Renders a curated feature table with mini sparkline bars.
 *
 * Props:
 *   features — { [label]: number } from display_features
 */

// Reasonable max values for normalizing each feature into a bar width.
// These are heuristics derived from typical image ranges.
const FEATURE_MAX = {
  'Edge Density':   0.5,
  'Texture Energy': 1.0,
  'Contrast':       5.0,
  'Homogeneity':    1.0,
  'Sharpness':      2000,
  'Pixel Entropy':  8.0,
  'Noise Level':    30.0,
  'Gradient Std':   0.2,
}

const DEFAULT_MAX = 1.0

export default function FeatureTable({ features }) {
  const entries = Object.entries(features)

  return (
    <div className="rounded-xl bg-surface-950/50 border border-slate-700/40 overflow-hidden">
      {entries.map(([key, value], i) => {
        const max = FEATURE_MAX[key] ?? DEFAULT_MAX
        const pct = Math.min((value / max) * 100, 100)

        return (
          <div
            key={key}
            className={[
              'feature-row px-4',
              i % 2 === 0 ? 'bg-transparent' : 'bg-slate-800/20',
            ].join(' ')}
          >
            {/* Feature name */}
            <span className="text-xs text-slate-400 w-36 shrink-0">{key}</span>

            {/* Mini bar */}
            <div className="flex-1 mx-4 h-1.5 bg-slate-800 rounded-full overflow-hidden hidden sm:block">
              <div
                className="h-full bg-slate-500 rounded-full"
                style={{ width: `${pct}%` }}
              />
            </div>

            {/* Value */}
            <span className="text-xs font-mono text-slate-300 w-16 text-right shrink-0">
              {formatValue(value)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function formatValue(v) {
  if (typeof v !== 'number') return String(v)
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  if (Math.abs(v) >= 10)   return v.toFixed(2)
  return v.toFixed(4)
}
