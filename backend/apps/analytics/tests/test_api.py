from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from apps.customers.models import Customer
from apps.products.models import Product
from apps.transactions.models import Transaction
from ml.services.result import ForecastResult


class AnalyticsAPITest(TestCase):
    def setUp(self):
        self.client = APIClient()

        customer = Customer.objects.create(
            customer_id=10001,
            country="United Kingdom",
        )

        product = Product.objects.create(
            stock_code="TEST001",
            description="Test product",
        )

        Transaction.objects.create(
            invoice_no="100001",
            customer=customer,
            product=product,
            quantity=2,
            invoice_date=timezone.make_aware(
                datetime(2024, 1, 1, 10, 0)
            ),
            unit_price=Decimal("10.00"),
            revenue=Decimal("20.00"),
        )

        Transaction.objects.create(
            invoice_no="100002",
            customer=customer,
            product=product,
            quantity=3,
            invoice_date=timezone.make_aware(
                datetime(2024, 1, 3, 10, 0)
            ),
            unit_price=Decimal("10.00"),
            revenue=Decimal("30.00"),
        )

    def test_sales_summary(self):
        response = self.client.get("/api/analytics/summary/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["transactions"], 2)
        self.assertEqual(response.data["customers"], 1)
        self.assertEqual(response.data["products"], 1)
        self.assertEqual(response.data["total_revenue"], "50.00")

    def test_daily_revenue_contains_zero_revenue_days(self):
        response = self.client.get("/api/analytics/revenue/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 3)

        self.assertEqual(
            response.data[0],
            {
                "date": "2024-01-01",
                "revenue": "20.00",
            },
        )

        self.assertEqual(
            response.data[1],
            {
                "date": "2024-01-02",
                "revenue": "0.00",
            },
        )

        self.assertEqual(
            response.data[2],
            {
                "date": "2024-01-03",
                "revenue": "30.00",
            },
        )

    @patch("apps.analytics.views.run_forecasting_service")
    def test_forecast(self, mock_run_forecasting_service):
        forecast = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03"]
                ),
                "Revenue": [0.0, 30.0],
                "PredictedRevenue": [25.0, 35.0],
            }
        )

        mock_run_forecasting_service.return_value = ForecastResult(
            forecast=forecast,
            metrics={
                "MAE": 5.0,
                "RMSE": 5.0,
                "WAPE": 0.1667,
                "NegativePredictions": 0,
                "MinPrediction": 25.0,
                "MaxPrediction": 35.0,
                "LargestAbsoluteError": 5.0,
            },
            validation={
                "Valid": True,
                "Rows": 2,
                "MissingColumns": [],
                "MissingPredictions": 0,
                "DuplicateDates": 0,
                "NegativePredictions": 0,
            },
            output_path=Path("forecast.csv"),
            training_rows=10,
            test_rows=2,
        )

        response = self.client.get("/api/analytics/forecast/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["forecast"]), 2)
        self.assertEqual(
            response.data["forecast"][0],
            {
                "Date": "2024-01-02",
                "Revenue": 0.0,
                "PredictedRevenue": 25.0,
            },
        )
        self.assertEqual(response.data["training_rows"], 10)
        self.assertEqual(response.data["test_rows"], 2)
        self.assertTrue(response.data["validation"]["Valid"])
        self.assertEqual(response.data["metrics"]["MAE"], 5.0)

        mock_run_forecasting_service.assert_called_once()