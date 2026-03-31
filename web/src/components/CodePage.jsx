import { useState } from 'react'
import CODE from './memberManagerCode.js'

export default function CodePage() {
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    navigator.clipboard.writeText(CODE).then(function () {
      setCopied(true)
      setTimeout(function () { setCopied(false) }, 2000)
    })
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0b0c10',
      color: '#e4e6eb',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
      padding: '24px'
    }}>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 16
        }}>
          <h1 style={{ fontSize: 20, fontWeight: 700 }}>
            Member<span style={{ color: '#818cf8' }}>Manager</span>
            <span style={{ fontSize: 13, color: '#5e616d', marginLeft: 10, fontWeight: 400 }}>
              — copy this code into a .html file
            </span>
          </h1>
          <button
            onClick={handleCopy}
            style={{
              padding: '8px 18px',
              fontSize: 13,
              fontWeight: 600,
              border: '1px solid ' + (copied ? 'rgba(34,211,153,.25)' : 'rgba(99,102,241,.3)'),
              borderRadius: 6,
              cursor: 'pointer',
              background: copied ? 'rgba(34,211,153,.08)' : 'rgba(99,102,241,.1)',
              color: copied ? '#34d399' : '#818cf8',
              transition: 'all 150ms'
            }}
          >
            {copied ? 'Copied!' : 'Copy to Clipboard'}
          </button>
        </div>
        <pre style={{
          background: '#111419',
          border: '1px solid #2a2e38',
          borderRadius: 10,
          padding: 20,
          overflow: 'auto',
          maxHeight: '80vh',
          fontSize: 12,
          lineHeight: 1.5,
          fontFamily: 'ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace',
          color: '#9a9da8',
          whiteSpace: 'pre',
          tabSize: 4
        }}>
          <code>{CODE}</code>
        </pre>
      </div>
    </div>
  )
}
