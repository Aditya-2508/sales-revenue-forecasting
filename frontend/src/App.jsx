import "./App.css";
import RevenueChart from "./components/RevenueChart";
import SummaryOverview from "./components/SummaryOverview";
import ForecastOverview from "./components/ForecastOverview";

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

        <SummaryOverview />
        <RevenueChart />
        <ForecastOverview />
      </main>
    </div>
  );
}

export default App;
