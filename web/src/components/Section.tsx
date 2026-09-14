import type { ReactNode } from "react";

interface SectionProps {
  id: string;
  index: string;
  title: string;
  lede: ReactNode;
  children: ReactNode;
}

export function Section({ id, index, title, lede, children }: SectionProps) {
  return (
    <section id={id} className="section" aria-labelledby={`${id}-title`}>
      <header className="section-head">
        <span className="section-index" aria-hidden="true">
          {index}
        </span>
        <div>
          <h2 id={`${id}-title`}>{title}</h2>
          <p className="lede">{lede}</p>
        </div>
      </header>
      {children}
    </section>
  );
}

export function Pending({ label }: { label: string }) {
  return (
    <p className="pending" role="status">
      {label}
    </p>
  );
}
