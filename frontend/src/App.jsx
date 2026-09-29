import { useState, useCallback } from 'react'
import UploadZone from './components/UploadZone'
import ResultPanel from './components/ResultPanel'
import Header from './components/Header'
import Footer from './components/Footer'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [state, setState] = useState({
    status: 'idle',   // idle | loading | success | error
    result: null,
    preview: null,
    error: null,
    fileName: null,
  })

  const handleFile = useCallback(async (file) => {
    // Build preview URL
    const preview = URL.createObjectURL(file)

    setState({
      status: 'loading',
      result: null,
      preview,
      error: null,
      fileName: file.name,
    })

    try {
      const form = new FormData()
      form.append('file', file)

      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setState(prev => ({ ...prev, status: 'success', result: data }))
    } catch (err) {
      setState(prev => ({
        ...prev,
        status: 'error',
        error: err.message,
      }))
    }
  }, [])

  const handleReset = useCallback(() => {
    setState({
      status: 'idle',
      result: null,
      preview: null,
      error: null,
      fileName: null,
    })
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      {/* Background glow */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-human-500/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -right-40 w-96 h-96 bg-ai-500/10 rounded-full blur-3xl" />
      </div>

      <Header />

      <main className="flex-1 relative z-10 container mx-auto max-w-5xl px-4 py-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* Left — Upload */}
          <div className="flex flex-col gap-6">
            <UploadZone
              onFile={handleFile}
              onReset={handleReset}
              preview={state.preview}
              loading={state.status === 'loading'}
              fileName={state.fileName}
            />

            {state.status === 'error' && (
              <div className="glass-card p-4 border-red-500/30 bg-red-950/30 animate-fade-up">
                <p className="text-red-400 text-sm font-medium">
                  ⚠️ {state.error}
                </p>
              </div>
            )}
          </div>

          {/* Right — Results */}
          <div>
            {state.status === 'loading' && <LoadingSkeleton />}
            {state.status === 'success' && state.result && (
              <ResultPanel result={state.result} />
            )}
            {state.status === 'idle' && <PlaceholderPanel />}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="glass-card p-6 space-y-4 animate-pulse-slow">
      <div className="h-6 bg-slate-700/60 rounded w-2/3" />
      <div className="h-12 bg-slate-700/60 rounded" />
      <div className="space-y-2">
        <div className="h-4 bg-slate-700/40 rounded" />
        <div className="h-4 bg-slate-700/40 rounded w-5/6" />
        <div className="h-4 bg-slate-700/40 rounded w-4/6" />
      </div>
      <p className="text-slate-500 text-sm text-center pt-2">Analysing image…</p>
    </div>
  )
}

function PlaceholderPanel() {
  return (
    <div className="glass-card p-8 flex flex-col items-center justify-center gap-4 min-h-[260px] text-center">
      <div className="text-5xl opacity-30">📊</div>
      <p className="text-slate-400 text-sm max-w-xs">
        Upload a drawing or sketch on the left to see the analysis results here.
      </p>
    </div>
  )
}
