from datetime import timedelta
from django.db.models import Sum
from django.db.models.functions import TruncDate
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.customers.models import Customer
from apps.products.models import Product
from apps.transactions.models import Transaction


class SalesSummaryView(APIView):
    def get(self, request):
        total_revenue = (
            Transaction.objects.aggregate(total=Sum("revenue"))["total"]
            or 0
        )

        return Response(
            {
                "transactions": Transaction.objects.count(),
                "customers": Customer.objects.count(),
                "products": Product.objects.count(),
                "total_revenue": f"{total_revenue:.2f}",
            }
        )


class DailyRevenueView(APIView):
    def get(self, request):
        daily_revenue = (
            Transaction.objects
            .annotate(date=TruncDate("invoice_date"))
            .values("date")
            .annotate(revenue=Sum("revenue"))
            .order_by("date")
        )

        revenue_by_date = {
            row["date"]: row["revenue"]
            for row in daily_revenue
        }

        first_date = min(revenue_by_date)
        last_date = max(revenue_by_date)

        results = []
        current_date = first_date

        while current_date <= last_date:
            revenue = revenue_by_date.get(current_date, 0)

            results.append(
                {
                    "date": current_date.isoformat(),
                    "revenue": f"{revenue:.2f}",
                }
            )

            current_date += timedelta(days=1)

        return Response(results)