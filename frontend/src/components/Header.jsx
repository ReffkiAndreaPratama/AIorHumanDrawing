export default function Header() {
  return (
    <header className="relative z-10 border-b border-slate-800/60 bg-surface-950/80 backdrop-blur-sm">
      <div className="container mx-auto max-w-5xl px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🎨</span>
          <div>
            <h1 className="text-base font-bold tracking-tight text-slate-100">
              AI vs Human Drawing Detector
            </h1>
            <p className="text-xs text-slate-500 hidden sm:block">
              Classical CV · GLCM · Edge · Frequency Analysis
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-human-500 animate-pulse" />
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">
            v1.0.0
          </span>
        </div>
      </div>
    </header>
  )
}
