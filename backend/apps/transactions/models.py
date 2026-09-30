from django.db import models

from apps.customers.models import Customer
from apps.products.models import Product


class Transaction(models.Model):
    invoice_no = models.CharField(max_length=20)
    customer = models.ForeignKey(
        Customer,
        on_delete=models.PROTECT,
        related_name="transactions",
    )
    product = models.ForeignKey(
        Product,
        on_delete=models.PROTECT,
        related_name="transactions",
    )
    quantity = models.IntegerField()
    invoice_date = models.DateTimeField()
    unit_price = models.DecimalField(
        max_digits=12,
        decimal_places=2,
    )
    revenue = models.DecimalField(
        max_digits=14,
        decimal_places=2,
    )

    def __str__(self):
        return f"{self.invoice_no} - {self.product.stock_code}"