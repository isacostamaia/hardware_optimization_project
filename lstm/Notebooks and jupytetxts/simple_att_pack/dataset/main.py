"""
Test purpose module
"""
from datetime import datetime, timedelta

import pandas as pd

from .settings import DB_CONNECTION_STRING
from .data_source.db_query import get_cpu_query

def main():
    """
    Main function
    """
    start = datetime.now()

    query = get_cpu_query(
        DB_CONNECTION_STRING,
        interval='hour',
        start_date=datetime.now() - timedelta(days=7),
        end_date=None,
        os='windows',
        machines_to_include=None,
        machines_to_exclude=None,
        limit=None
    )
    records = query.all()
    df = pd.DataFrame(records, columns=['date', 'hostname', 'os', 'cpu'])
    print(df.info())
    print()
    print(df.head(5))
    print()

    end = datetime.now()
    duration = (end - start).total_seconds()
    print('Duration1: {}s'.format(duration))

# Run as program
if __name__ == '__main__':
    main()
