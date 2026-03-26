import { useEffect, useRef, useState } from 'react'

function AnimatedNumber({ value, suffix = '', duration = 1500 }) {
  const [display, setDisplay] = useState(0)
  const ref = useRef(null)
  const started = useRef(false)

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started.current) {
        started.current = true
        const start = performance.now()
        const animate = (now) => {
          const progress = Math.min((now - start) / duration, 1)
          const eased = 1 - Math.pow(1 - progress, 3)
          setDisplay(eased * value)
          if (progress < 1) requestAnimationFrame(animate)
        }
        requestAnimationFrame(animate)
      }
    }, { threshold: 0.5 })

    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [value, duration])

  const formatted = Number.isInteger(value) ? Math.round(display) : display.toFixed(1)
  return <span ref={ref}>{formatted}{suffix}</span>
}

const stats = [
  { value: 397, suffix: 'B', label: 'Largest model', sub: 'Qwen3.5-397B on any Mac' },
  { value: 13, suffix: 'x', label: 'RAM reduction', sub: '20GB → 2GB for 35B model' },
  { value: 6.5, suffix: ' t/s', label: 'Decode speed', sub: '35B K=4 on Apple M4' },
  { value: 5.5, suffix: 'x', label: 'Prefill speedup', sub: 'Custom Metal GPU kernels' },
]

const rows = [
  { label: 'Model', col1: 'Qwen3.5-35B', col2: 'Qwen3.5-122B', col3: 'Qwen3.5-397B' },
  { label: 'Parameters', col1: '35B (3B active)', col2: '122B (10B active)', col3: '397B (17B active)' },
  { label: 'Experts', col1: '256 total, 8 active', col2: '256 total, 8 active', col3: '512 total, 10 active' },
  { label: 'Disk', col1: '20 GB', col2: '70 GB', col3: '224 GB' },
  { label: 'Standard RAM', col1: '20 GB', col2: '70 GB', col3: '224 GB' },
  { label: 'Kandiga RAM', col1: '~2 GB', col2: '~4 GB', col3: '~8 GB', highlight: true },
  { label: 'Min. Mac', col1: '8 GB', col2: '16 GB', col3: '24 GB', highlight: true },
]

const models = [
  {
    name: 'Qwen3.5-35B',
    sub: '3B active · 256 experts',
    specs: [
      { label: 'Disk', value: '20 GB' },
      { label: 'RAM', value: '~2 GB', good: true },
      { label: 'Speed', value: '3.4–6.5 t/s' },
      { label: 'Min. Mac', value: '8 GB' },
    ],
  },
  {
    name: 'Qwen3.5-122B',
    sub: '10B active · 256 experts',
    specs: [
      { label: 'Disk', value: '70 GB' },
      { label: 'RAM', value: '~4 GB', good: true },
      { label: 'Speed', value: '~2 t/s' },
      { label: 'Min. Mac', value: '16 GB' },
    ],
  },
  {
    name: 'Qwen3.5-397B',
    sub: '17B active · 512 experts',
    featured: true,
    specs: [
      { label: 'Disk', value: '224 GB' },
      { label: 'RAM', value: '~8 GB', good: true },
      { label: 'Speed', value: '~1 t/s' },
      { label: 'Min. Mac', value: '24 GB' },
    ],
  },
]

export default function Stats() {
  return (
    <section className="py-4">
      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
        {stats.map((s, i) => (
          <div key={i} className="feature-card text-center">
            <div className="text-2xl md:text-3xl font-bold text-[var(--text-bright)] mb-1 tracking-tight">
              <AnimatedNumber value={s.value} suffix={s.suffix} />
            </div>
            <div className="text-sm text-[var(--text)]">{s.label}</div>
            <div className="text-xs text-[var(--text-muted)]">{s.sub}</div>
          </div>
        ))}
      </div>

      <h3 className="text-lg font-semibold text-[var(--text-bright)] mb-2">Supported models</h3>
      <p className="text-sm text-[var(--text-muted)] mb-5">Pick a model that fits your Mac. Kandiga handles the rest.</p>

      {/* Mobile: 3 model cards stacked */}
      <div className="flex flex-col gap-4 md:hidden">
        {models.map((m, i) => (
          <div key={i} className={`rounded-xl p-4 border ${m.featured
            ? 'bg-[var(--card)] border-[var(--cyan)]/30'
            : 'bg-[var(--surface)] border-[var(--card-border)]'}`}>
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className={`font-semibold text-sm ${m.featured ? 'text-[var(--cyan)]' : 'text-[var(--text-bright)]'}`}>{m.name}</div>
                <div className="text-[11px] text-[var(--text-muted)]">{m.sub}</div>
              </div>
              {m.featured && (
                <span className="text-[10px] font-medium text-[var(--cyan)] bg-[var(--cyan-dim)] px-2 py-0.5 rounded-full">Flagship</span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
              {m.specs.map((s, j) => (
                <div key={j} className="flex justify-between items-baseline">
                  <span className="text-[11px] text-[var(--text-muted)]">{s.label}</span>
                  <span className={`text-[13px] font-medium ${
                    s.good ? 'text-[var(--cyan)]' : s.bad ? 'text-red-400/80' : 'text-[var(--text)]'
                  }`}>{s.value}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Desktop: table */}
      <div className="hidden md:block feature-card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--card-border)]">
              <th className="pb-3 text-left text-[var(--text-muted)] font-medium"></th>
              <th className="pb-3 text-left text-[var(--text-bright)] font-medium">Qwen3.5-35B</th>
              <th className="pb-3 text-left text-[var(--text-bright)] font-medium">Qwen3.5-122B</th>
              <th className="pb-3 text-left text-[var(--cyan)] font-medium">Qwen3.5-397B</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-[var(--card-border)]/30 last:border-0">
                <td className="py-3 text-[var(--text-muted)]">{row.label}</td>
                <td className="py-3 text-[var(--text)]">{row.col1}</td>
                <td className="py-3 text-[var(--text)]">{row.col2}</td>
                <td className={`py-3 font-medium ${row.highlight ? 'text-[var(--cyan)]' : 'text-[var(--text-bright)]'}`}>{row.col3}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
