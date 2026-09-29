import { useEffect, useRef, useState } from 'react'

/**
 * Animated horizontal confidence bar.
 *
 * Props:
 *   label      — e.g. "✏️ Human"
 *   value      — 0.0 to 1.0
 *   colorClass — Tailwind bg class e.g. "bg-human-500"
 */
export default function ConfidenceBar({ label, value, colorClass }) {
  const [width, setWidth] = useState(0)
  const rafRef = useRef(null)

  useEffect(() => {
    // Small delay so the CSS transition plays on mount
    const id = setTimeout(() => setWidth(value * 100), 60)
    return () => { clearTimeout(id); cancelAnimationFrame(rafRef.current) }
  }, [value])

  const pct = (value * 100).toFixed(1)

  return (
    <div className="flex items-center gap-3">
      {/* Label */}
      <span className="text-sm text-slate-400 w-24 shrink-0">{label}</span>

      {/* Bar track */}
      <div className="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden"
           role="progressbar"
           aria-valuenow={Math.round(value * 100)}
           aria-valuemin={0}
           aria-valuemax={100}
           aria-label={`${label}: ${pct}%`}>
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${colorClass}`}
          style={{ width: `${width}%` }}
        />
      </div>

      {/* Percentage */}
      <span className="text-sm font-mono text-slate-300 w-14 text-right shrink-0">
        {pct}%
      </span>
    </div>
  )
}
