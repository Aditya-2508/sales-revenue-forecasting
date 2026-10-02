import pandas as pd

from apps.transactions.models import Transaction


def transactions_to_dataframe() -> pd.DataFrame:
    """
    Convert Django transactions into the raw-data schema
    expected by the ML forecasting service.
    """

    transactions = (
        Transaction.objects
        .select_related("customer", "product")
        .order_by("id")
    )

    records = [
        {
            "InvoiceNo": transaction.invoice_no,
            "StockCode": transaction.product.stock_code,
            "Description": transaction.product.description,
            "Quantity": transaction.quantity,
            "InvoiceDate": transaction.invoice_date,
            "UnitPrice": float(transaction.unit_price),
            "CustomerID": transaction.customer.customer_id,
            "Country": transaction.customer.country,
        }
        for transaction in transactions
    ]

    df = pd.DataFrame(records)

    if not df.empty:
        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"],
            utc=True,
        ).dt.tz_localize(None)

    return df