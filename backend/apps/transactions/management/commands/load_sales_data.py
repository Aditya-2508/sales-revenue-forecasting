from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from apps.customers.models import Customer
from apps.products.models import Product
from apps.transactions.models import Transaction
from ml.preprocessing.cleaning import clean_sales_data


class Command(BaseCommand):
    help = "Load cleaned sales data into the Django database."

    def handle(self, *args, **options):
        if (
            Customer.objects.exists()
            or Product.objects.exists()
            or Transaction.objects.exists()
        ):
            raise CommandError(
                "The database already contains sales data. "
                "Clear the incomplete load before running this command."
            )

        project_root = Path(__file__).resolve().parents[5]
        raw_file = project_root / "data" / "raw" / "Online Retail.xlsx"

        self.stdout.write(f"Loading dataset from: {raw_file}")

        df = pd.read_excel(raw_file)
        cleaned_df = clean_sales_data(df)

        self.stdout.write(
            f"Cleaned transaction rows: {len(cleaned_df)}"
        )

        cleaned_df["InvoiceDate"] = pd.to_datetime(
            cleaned_df["InvoiceDate"]
        )

        if cleaned_df["InvoiceDate"].dt.tz is None:
            cleaned_df["InvoiceDate"] = (
                cleaned_df["InvoiceDate"]
                .dt.tz_localize(timezone.get_current_timezone())
            )

        customer_df = (
            cleaned_df[["CustomerID", "Country"]]
            .drop_duplicates(subset=["CustomerID"])
        )

        product_df = (
            cleaned_df[["StockCode", "Description"]]
            .drop_duplicates(subset=["StockCode"])
        )

        with transaction.atomic():
            Customer.objects.bulk_create(
                [
                    Customer(
                        customer_id=int(row.CustomerID),
                        country=str(row.Country),
                    )
                    for row in customer_df.itertuples(index=False)
                ],
                batch_size=1000,
            )

            Product.objects.bulk_create(
                [
                    Product(
                        stock_code=str(row.StockCode),
                        description=(
                            str(row.Description)
                            if pd.notna(row.Description)
                            else ""
                        ),
                    )
                    for row in product_df.itertuples(index=False)
                ],
                batch_size=1000,
            )

            customers = {
                customer.customer_id: customer
                for customer in Customer.objects.all()
            }

            products = {
                product.stock_code: product
                for product in Product.objects.all()
            }

            transactions = [
                Transaction(
                    invoice_no=str(row.InvoiceNo),
                    customer=customers[int(row.CustomerID)],
                    product=products[str(row.StockCode)],
                    quantity=int(row.Quantity),
                    invoice_date=row.InvoiceDate,
                    unit_price=row.UnitPrice,
                    revenue=row.Revenue,
                )
                for row in cleaned_df.itertuples(index=False)
            ]

            Transaction.objects.bulk_create(
                transactions,
                batch_size=5000,
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Loaded {Transaction.objects.count()} transactions, "
                f"{Customer.objects.count()} customers, "
                f"{Product.objects.count()} products."
            )
        )
