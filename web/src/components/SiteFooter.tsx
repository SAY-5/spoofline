const REPO = "https://github.com/SAY-5/spoofline";

export function SiteFooter({ commit }: { commit: string | null }) {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <p className="footer-mark">spoofline</p>
        <nav aria-label="Project links">
          <ul className="footer-links">
            <li>
              <a href={REPO}>Repository</a>
            </li>
            <li>
              <a href={`${REPO}/blob/main/README.md`}>README</a>
            </li>
            <li>
              <a href={`${REPO}/blob/main/docs/ARCHITECTURE.md`}>ARCHITECTURE.md</a>
            </li>
            <li>
              <a href={`${REPO}/blob/main/web/scripts/export.py`}>Export script</a>
            </li>
          </ul>
        </nav>
        <p className="footer-meta">
          {commit ? (
            <>
              Weights trained from commit <a href={`${REPO}/commit/${commit}`}>{commit.slice(0, 7)}</a>.
            </>
          ) : (
            "Loading run metadata."
          )}
        </p>
      </div>
    </footer>
  );
}
