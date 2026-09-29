import { useRef, useState, useCallback } from 'react'

const ACCEPTED = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp']
const MAX_MB = 10

export default function UploadZone({ onFile, onReset, preview, loading, fileName }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [localError, setLocalError] = useState(null)

  const validate = (file) => {
    if (!ACCEPTED.includes(file.type)) {
      return `Unsupported format: ${file.type}. Use JPEG, PNG, BMP, or WebP.`
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      return `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_MB} MB.`
    }
    return null
  }

  const handleFile = useCallback((file) => {
    const err = validate(file)
    if (err) { setLocalError(err); return }
    setLocalError(null)
    onFile(file)
  }, [onFile])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    e.target.value = ''   // allow re-selecting same file
  }

  // ── Keyboard / paste support ────────────────────────────────────────────
  const onPaste = useCallback((e) => {
    const items = e.clipboardData?.items
    if (!items) return
    for (const item of items) {
      if (item.kind === 'file' && ACCEPTED.includes(item.type)) {
        handleFile(item.getAsFile())
        return
      }
    }
  }, [handleFile])

  const hasImage = Boolean(preview)

  return (
    <div className="flex flex-col gap-3" onPaste={onPaste} tabIndex={0}
         aria-label="Upload zone">
      {/* Drop area */}
      <div
        onClick={() => !hasImage && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        aria-label={hasImage ? 'Uploaded image preview' : 'Click or drag to upload image'}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click() }}
        className={[
          'relative rounded-2xl border-2 transition-all duration-200 overflow-hidden',
          'min-h-[280px] flex flex-col items-center justify-center',
          hasImage
            ? 'border-slate-700/50 cursor-default'
            : 'border-dashed cursor-pointer',
          dragOver
            ? 'border-human-500 bg-human-500/5'
            : hasImage
              ? 'border-slate-700/50'
              : 'border-slate-600 hover:border-slate-400 bg-surface-900/40 hover:bg-surface-900/70',
        ].join(' ')}
      >
        {hasImage ? (
          <>
            <img
              src={preview}
              alt="Uploaded drawing"
              className="w-full h-full object-contain max-h-[320px] p-2"
            />
            {loading && (
              <div className="absolute inset-0 bg-surface-950/70 flex flex-col
                              items-center justify-center gap-3 backdrop-blur-sm">
                <Spinner />
                <p className="text-slate-300 text-sm font-medium">
                  Extracting features…
                </p>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center gap-4 p-8 text-center
                          pointer-events-none select-none">
            <div className={[
              'text-5xl transition-transform duration-200',
              dragOver ? 'scale-110' : 'scale-100',
            ].join(' ')}>
              {dragOver ? '📂' : '🖼️'}
            </div>
            <div>
              <p className="text-slate-300 font-medium text-sm">
                Drop your drawing here
              </p>
              <p className="text-slate-500 text-xs mt-1">
                or click to browse · paste from clipboard
              </p>
            </div>
            <p className="text-slate-600 text-xs font-mono">
              JPEG · PNG · BMP · WebP · max {MAX_MB} MB
            </p>
          </div>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        onChange={onInputChange}
        className="hidden"
        aria-hidden="true"
      />

      {/* File name + actions */}
      {hasImage && (
        <div className="flex items-center justify-between gap-2 px-1 animate-fade-up">
          <span className="text-slate-400 text-xs truncate font-mono max-w-[200px]"
                title={fileName}>
            📎 {fileName}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => inputRef.current?.click()}
              disabled={loading}
              className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700
                         text-slate-300 transition disabled:opacity-50"
            >
              Change
            </button>
            <button
              onClick={onReset}
              disabled={loading}
              className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-red-900/50
                         text-slate-300 hover:text-red-400 transition disabled:opacity-50"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {localError && (
        <p className="text-red-400 text-xs px-1">⚠️ {localError}</p>
      )}
    </div>
  )
}

function Spinner() {
  return (
    <svg className="w-8 h-8 animate-spin text-human-500"
         viewBox="0 0 24 24" fill="none"
         aria-label="Loading" role="img">
      <circle className="opacity-25" cx="12" cy="12" r="10"
              stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}
