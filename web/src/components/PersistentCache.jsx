export default function PersistentCache() {
  return (
    <section>
      <h2 className="text-center text-2xl md:text-3xl font-semibold text-[var(--text-bright)] mb-4 tracking-tight">
        Conversations that never slow down
      </h2>
      <p className="text-center text-[var(--text-muted)] mb-10 max-w-2xl mx-auto">
        Most local LLMs re-read your entire conversation on every message.
        Turn 30 re-processes turns 1-29 from scratch. Kandiga keeps the model's
        understanding in memory — follow-ups only process the new message.
      </p>

      {/* Comparison */}
      <div className="grid md:grid-cols-2 gap-4 max-w-3xl mx-auto mb-10">
        {/* Without */}
        <div className="bg-[var(--surface)] border border-[var(--card-border)] rounded-xl p-5">
          <div className="text-sm font-semibold text-red-400/80 mb-3">Without persistent cache</div>
          <div className="space-y-2 font-mono text-xs">
            {[
              { turn: 'Turn 1', time: '8s', tokens: '100 tokens' },
              { turn: 'Turn 5', time: '25s', tokens: '600 tokens' },
              { turn: 'Turn 10', time: '50s', tokens: '1500 tokens' },
              { turn: 'Turn 30', time: '2min+', tokens: '5000 tokens' },
            ].map((r, i) => (
              <div key={i} className="grid grid-cols-[1fr_60px_100px] text-[var(--text-muted)]">
                <span>{r.turn}</span>
                <span className="text-red-400/80 text-right">{r.time}</span>
                <span className="text-[11px] text-right">{r.tokens}</span>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-[var(--text-muted)] mt-3">Re-reads everything every turn</p>
        </div>

        {/* With */}
        <div className="bg-[var(--card)] border border-[var(--cyan)]/30 rounded-xl p-5">
          <div className="text-sm font-semibold text-[var(--cyan)] mb-3">With persistent KV cache</div>
          <div className="space-y-2 font-mono text-xs">
            {[
              { turn: 'Turn 1', time: '8s', tokens: '100 tokens' },
              { turn: 'Turn 5', time: '3s', tokens: '20 new tokens' },
              { turn: 'Turn 10', time: '3s', tokens: '25 new tokens' },
              { turn: 'Turn 30', time: '3s', tokens: '20 new tokens' },
            ].map((r, i) => (
              <div key={i} className="grid grid-cols-[1fr_60px_100px] text-[var(--text-muted)]">
                <span>{r.turn}</span>
                <span className="text-[var(--cyan)] font-medium text-right">{r.time}</span>
                <span className="text-[11px] text-right">{r.tokens}</span>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-[var(--text-muted)] mt-3">Only reads new messages</p>
        </div>
      </div>

      {/* Explainer cards */}
      <div className="grid sm:grid-cols-3 gap-4 text-center max-w-3xl mx-auto">
        {[
          { value: '3-4s', label: 'Every follow-up', desc: 'Constant regardless of history' },
          { value: '0 tokens', label: 'Re-processed', desc: 'Previous turns stay cached' },
          { value: '32K+', label: 'Context window', desc: 'With TurboQuant compression' },
        ].map((item, i) => (
          <div key={i} className="feature-card text-center">
            <div className="text-lg font-bold text-[var(--cyan)] mb-1">{item.value}</div>
            <div className="text-xs sm:text-sm text-[var(--text-bright)] mb-1">{item.label}</div>
            <div className="text-[10px] sm:text-xs text-[var(--text-muted)]">{item.desc}</div>
          </div>
        ))}
      </div>
    </section>
  )
}
