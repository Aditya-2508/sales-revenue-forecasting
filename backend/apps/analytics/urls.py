from django.urls import path

from apps.analytics.views import DailyRevenueView, SalesSummaryView


urlpatterns = [
    path("summary/", SalesSummaryView.as_view(), name="sales-summary"),
    path("revenue/", DailyRevenueView.as_view(), name="daily-revenue"),
]
