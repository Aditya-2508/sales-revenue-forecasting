import "./App.css";

function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            SR
          </span>

          <div>
            <p className="brand-name">Sales & Revenue</p>
            <p className="brand-subtitle">Forecasting Platform</p>
          </div>
        </div>

        <nav className="primary-navigation" aria-label="Primary navigation">
          <a
            className="navigation-link navigation-link-active"
            href="#overview"
          >
            Overview
          </a>
          <a className="navigation-link" href="#sales">
            Sales
          </a>
          <a className="navigation-link" href="#forecasting">
            Forecasting
          </a>
        </nav>
      </header>

      <main className="app-content">
        <section className="page-introduction" id="overview">
          <p className="section-label">Business intelligence</p>
          <h1>Sales & revenue overview</h1>
          <p className="page-description">
            Review historical sales activity and model-based revenue forecasts
            from the available transaction data.
          </p>
        </section>

        <section className="empty-state" aria-labelledby="dashboard-heading">
          <p className="section-label">Dashboard</p>
          <h2 id="dashboard-heading">Data connection coming next</h2>
          <p>
            The dashboard will display verified sales and forecasting results
            from the Django API.
          </p>
        </section>
      </main>
    </div>
  );
}

export default App;
