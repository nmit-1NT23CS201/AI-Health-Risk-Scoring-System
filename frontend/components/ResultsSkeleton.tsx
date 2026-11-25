export default function ResultsSkeleton() {
  return (
    <div className="stack" aria-busy aria-live="polite">
      <span className="sr-only">Calculating estimates…</span>
      {[0, 1, 2].map((i) => (
        <div className="glass disease-card" key={i}>
          <div className="skeleton sk-title" />
          <div className="skeleton sk-big" />
          <div className="skeleton sk-bar" />
          {[0, 1, 2, 3].map((r) => (
            <div className="sk-row" key={r}>
              <div className="skeleton sk-line" />
              <div className="skeleton sk-line" style={{ width: `${40 + ((r * 17 + i * 11) % 50)}%` }} />
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
