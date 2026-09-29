import ConfidenceBar from './ConfidenceBar'
import FeatureTable from './FeatureTable'

const LABEL_CONFIG = {
  Human: {
    emoji: '✏️',
    colorClass: 'text-human-400',
    bgClass: 'bg-human-500/10 border-human-500/30',
    barColor: 'human',
  },
  AI: {
    emoji: '🤖',
    colorClass: 'text-ai-400',
    bgClass: 'bg-ai-500/10 border-ai-500/30',
    barColor: 'ai',
  },
  Uncertain: {
    emoji: '❓',
    colorClass: 'text-uncertain-400',
    bgClass: 'bg-uncertain-500/10 border-uncertain-500/30',
    barColor: 'uncertain',
  },
}

export default function ResultPanel({ result }) {
  const cfg = LABEL_CONFIG[result.label] ?? LABEL_CONFIG.Uncertain

  return (
    <div className="glass-card p-6 flex flex-col gap-6 animate-fade-up">

      {/* ── Prediction badge ─────────────────────────────── */}
      <div className={`rounded-xl border p-4 ${cfg.bgClass}`}>
        <div className="flex items-center gap-3">
          <span className="text-3xl" role="img" aria-label={result.label}>
            {cfg.emoji}
          </span>
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-widest font-medium">
              Prediction
            </p>
            <p className={`text-2xl font-bold ${cfg.colorClass}`}>
              {result.label === 'Human' ? 'Human Drawing' : result.label === 'AI' ? 'AI Generated' : 'Uncertain'}
            </p>
          </div>
          <div className="ml-auto text-right">
            <p className="text-xs text-slate-500 uppercase tracking-widest font-medium">
              Confidence
            </p>
            <p className={`text-2xl font-bold font-mono ${cfg.colorClass}`}>
              {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* ── Probability bars ─────────────────────────────── */}
      <div className="flex flex-col gap-3">
        <p className="text-xs text-slate-500 uppercase tracking-widest font-medium">
          Class Probabilities
        </p>
        <ConfidenceBar
          label="✏️ Human"
          value={result.human_prob}
          colorClass="bg-human-500"
        />
        <ConfidenceBar
          label="🤖 AI"
          value={result.ai_prob}
          colorClass="bg-ai-500"
        />
      </div>

      {/* ── Feature analysis ─────────────────────────────── */}
      {result.display_features && Object.keys(result.display_features).length > 0 && (
        <div className="flex flex-col gap-3">
          <p className="text-xs text-slate-500 uppercase tracking-widest font-medium">
            Image Analysis
          </p>
          <FeatureTable features={result.display_features} />
        </div>
      )}

      {/* ── Meta ─────────────────────────────────────────── */}
      <div className="flex items-center justify-between pt-1 border-t border-slate-700/40">
        <span className="text-xs text-slate-600 font-mono">
          {result.inference_time_ms} ms
        </span>
        <span className="text-xs text-slate-600">
          SVM + Random Forest ensemble
        </span>
      </div>

      {/* ── Disclaimer ───────────────────────────────────── */}
      <p className="text-xs text-slate-600 leading-relaxed border-t border-slate-700/40 pt-3">
        ℹ️ {result.disclaimer}
      </p>
    </div>
  )
}
