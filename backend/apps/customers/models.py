from django.db import models


class Customer(models.Model):
    customer_id = models.PositiveIntegerField(unique=True)
    country = models.CharField(max_length=100)

    def __str__(self):
        return str(self.customer_id)