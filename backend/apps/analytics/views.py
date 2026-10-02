from datetime import timedelta
from django.db.models import Sum
from django.db.models.functions import TruncDate
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.customers.models import Customer
from apps.products.models import Product
from apps.transactions.models import Transaction

from pathlib import Path

from django.conf import settings

from apps.analytics.services.forecast_adapter import transactions_to_dataframe
from ml.services.forecasting import run_forecasting_service
from ml.services.response import build_forecast_response


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
    

class ForecastView(APIView):
    def get(self, request):
        raw_df = transactions_to_dataframe()

        output_path = (
            Path(settings.BASE_DIR).parent
            / "data"
            / "forecasts"
            / "latest_forecast.csv"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = run_forecasting_service(
            raw_df=raw_df,
            output_path=output_path,
        )

        return Response(build_forecast_response(result))