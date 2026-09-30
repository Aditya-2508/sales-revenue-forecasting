from datetime import datetime
from django.utils import timezone
from decimal import Decimal

from django.test import TestCase
from rest_framework.test import APIClient

from apps.customers.models import Customer
from apps.products.models import Product
from apps.transactions.models import Transaction


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
            invoice_date=timezone.make_aware(datetime(2024, 1, 1, 10, 0)),
            unit_price=Decimal("10.00"),
            revenue=Decimal("20.00"),
        )

        Transaction.objects.create(
            invoice_no="100002",
            customer=customer,
            product=product,
            quantity=3,
            invoice_date=timezone.make_aware(datetime(2024, 1, 3, 10, 0)),
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

        self.assertEqual(response.data[0], {
            "date": "2024-01-01",
            "revenue": "20.00",
        })

        self.assertEqual(response.data[1], {
            "date": "2024-01-02",
            "revenue": "0.00",
        })

        self.assertEqual(response.data[2], {
            "date": "2024-01-03",
            "revenue": "30.00",
        })
