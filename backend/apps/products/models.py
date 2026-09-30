from django.db import models


class Product(models.Model):
    stock_code = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.stock_code