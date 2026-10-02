import { useEffect, useState } from "react";
import { fetchSalesSummary } from "../services/api";

function formatNumber(value) {
  return new Intl.NumberFormat("en-IN").format(value);
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function SummaryOverview() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    fetchSalesSummary()
      .then((data) => {
        if (isMounted) {
          setSummary(data);
        }
      })
      .catch((requestError) => {
        if (isMounted) {
          setError(requestError.message);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  if (error) {
    return (
      <section className="data-section" aria-labelledby="summary-error-heading">
        <p className="section-label">Sales summary</p>
        <h2 id="summary-error-heading">Unable to load sales data</h2>
        <p className="data-message">{error}</p>
      </section>
    );
  }

  if (!summary) {
    return (
      <section className="data-section" aria-live="polite">
        <p className="section-label">Sales summary</p>
        <h2>Loading sales data...</h2>
      </section>
    );
  }

  return (
    <section className="summary-section" aria-labelledby="summary-heading">
      <div className="section-heading">
        <div>
          <p className="section-label">Sales summary</p>
          <h2 id="summary-heading">Recorded business activity</h2>
        </div>
      </div>

      <div className="summary-grid">
        <article className="summary-item">
          <p className="summary-label">Total revenue</p>
          <p className="summary-value">
            {formatCurrency(summary.total_revenue)}
          </p>
        </article>

        <article className="summary-item">
          <p className="summary-label">Transactions</p>
          <p className="summary-value">{formatNumber(summary.transactions)}</p>
        </article>

        <article className="summary-item">
          <p className="summary-label">Customers</p>
          <p className="summary-value">{formatNumber(summary.customers)}</p>
        </article>

        <article className="summary-item">
          <p className="summary-label">Products</p>
          <p className="summary-value">{formatNumber(summary.products)}</p>
        </article>
      </div>
    </section>
  );
}

export default SummaryOverview;
