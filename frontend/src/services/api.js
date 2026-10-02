const API_BASE_URL = "/api";

async function request(endpoint) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);

  if (!response.ok) {
    throw new Error(`API request failed with status ${response.status}`);
  }

  return response.json();
}

export function fetchSalesSummary() {
  return request("/analytics/summary/");
}

export function fetchDailyRevenue() {
  return request("/analytics/revenue/");
}