import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchForecast } from "../services/api";

function formatCurrency(value) {
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 0,
  }).format(Number(value));
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(new Date(`${value}T00:00:00`));
}

function ForecastOverview() {
  const [forecastData, setForecastData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    fetchForecast()
      .then((data) => {
        if (isMounted) {
          setForecastData(data.forecast);
          setMetrics(data.metrics);
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
      <section
        className="data-section"
        aria-labelledby="forecast-error-heading"
      >
        <p className="section-label">Revenue forecast</p>
        <h2 id="forecast-error-heading">Unable to load forecast data</h2>
        <p className="data-message">{error}</p>
      </section>
    );
  }

  if (!forecastData || !metrics) {
    return (
      <section className="data-section" aria-live="polite">
        <p className="section-label">Revenue forecast</p>
        <h2>Loading forecast data...</h2>
      </section>
    );
  }

  const chartData = forecastData.map((item) => ({
    ...item,
    ActualRevenue: Number(item.Revenue),
    ForecastRevenue: Number(item.PredictedRevenue),
  }));

  return (
    <section className="forecast-section" aria-labelledby="forecast-heading">
      <div className="section-heading">
        <div>
          <p className="section-label">Model output</p>
          <h2 id="forecast-heading">Forecast performance</h2>
        </div>
        <p className="chart-meta">
          {forecastData.length} evaluation days ·{" "}
          {formatDate(forecastData[0].Date)} to{" "}
          {formatDate(forecastData[forecastData.length - 1].Date)}
        </p>
      </div>

      <div
        className="forecast-metrics"
        aria-label="Forecast evaluation metrics"
      >
        <article className="forecast-metric">
          <p className="summary-label">MAE</p>
          <p className="forecast-metric-value">{formatCurrency(metrics.MAE)}</p>
        </article>

        <article className="forecast-metric">
          <p className="summary-label">RMSE</p>
          <p className="forecast-metric-value">
            {formatCurrency(metrics.RMSE)}
          </p>
        </article>

        <article className="forecast-metric">
          <p className="summary-label">WAPE</p>
          <p className="forecast-metric-value">
            {(Number(metrics.WAPE) * 100).toFixed(1)}%
          </p>
        </article>
      </div>

      <div className="forecast-chart">
        <ResponsiveContainer width="100%" height={360}>
          <LineChart
            data={chartData}
            margin={{ top: 12, right: 16, left: 8, bottom: 8 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke="var(--color-border)"
            />

            <XAxis
              dataKey="Date"
              tickFormatter={(value) =>
                new Intl.DateTimeFormat("en-GB", {
                  month: "short",
                  year: "2-digit",
                }).format(new Date(`${value}T00:00:00`))
              }
              minTickGap={36}
              tick={{ fill: "var(--color-text-muted)", fontSize: 12 }}
              axisLine={{ stroke: "var(--color-border)" }}
              tickLine={false}
            />

            <YAxis
              tickFormatter={(value) => `£${Math.round(value / 1000)}k`}
              width={52}
              tick={{ fill: "var(--color-text-muted)", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />

            <Tooltip
              labelFormatter={formatDate}
              formatter={(value, name) => [
                formatCurrency(value),
                name === "Actual revenue" ? "Actual revenue" : "Forecast",
              ]}
              contentStyle={{
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                boxShadow: "var(--shadow-medium)",
              }}
            />

            <Legend />

            <Line
              type="monotone"
              dataKey="ActualRevenue"
              name="Actual revenue"
              stroke="var(--color-text)"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />

            <Line
              type="monotone"
              dataKey="ForecastRevenue"
              name="Forecast"
              stroke="var(--color-accent)"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

export default ForecastOverview;
