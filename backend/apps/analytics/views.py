from django.db.models import Sum
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
