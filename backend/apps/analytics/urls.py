from django.urls import path

from apps.analytics.views import SalesSummaryView


urlpatterns = [
    path("summary/", SalesSummaryView.as_view(), name="sales-summary"),
]
