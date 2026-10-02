import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchDailyRevenue } from "../services/api";

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

function RevenueChart() {
  const [revenueData, setRevenueData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    fetchDailyRevenue()
      .then((data) => {
        if (isMounted) {
          setRevenueData(data);
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
      <section className="data-section" aria-labelledby="revenue-error-heading">
        <p className="section-label">Revenue trend</p>
        <h2 id="revenue-error-heading">Unable to load revenue data</h2>
        <p className="data-message">{error}</p>
      </section>
    );
  }

  if (!revenueData) {
    return (
      <section className="data-section" aria-live="polite">
        <p className="section-label">Revenue trend</p>
        <h2>Loading revenue data...</h2>
      </section>
    );
  }

  return (
    <section className="revenue-section" aria-labelledby="revenue-heading">
      <div className="section-heading">
        <div>
          <p className="section-label">Historical performance</p>
          <h2 id="revenue-heading">Daily revenue</h2>
        </div>
        <p className="chart-meta">
          {revenueData.length} days · {formatDate(revenueData[0].date)} to{" "}
          {formatDate(revenueData[revenueData.length - 1].date)}
        </p>
      </div>

      <div className="revenue-chart">
        <ResponsiveContainer width="100%" height={360}>
          <LineChart
            data={revenueData}
            margin={{ top: 12, right: 16, left: 8, bottom: 8 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke="var(--color-border)"
            />
            <XAxis
              dataKey="date"
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
              formatter={(value) => [formatCurrency(value), "Revenue"]}
              contentStyle={{
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                boxShadow: "var(--shadow-medium)",
              }}
            />
            <Line
              type="monotone"
              dataKey="revenue"
              stroke="var(--color-primary)"
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

export default RevenueChart;
