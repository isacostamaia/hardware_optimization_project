"""
Define the SQL queries for Nagios DB data fetch
"""
import warnings

from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import asc, func, case, text, exc, Integer, Float
from sqlalchemy.sql.expression import cast

# Ignore not legit MySQL Float conversion warning
warnings.filterwarnings('ignore', category=exc.SAWarning)

def get_session(connection_string):
    """
    Get an SQL DB session + base schema map for use with SQLA ORM
    connection_string: DB connection string,
    eg. mysql://user:password@host/db_name
    Return (session, base.classes)
    """
    # Map DB schema to DB engine
    base = automap_base()
    engine = create_engine(connection_string, pool_recycle=3600)
    base.prepare(engine, reflect=True)
    # DB tables are accessible within ORM session through base.classes.<table name>
    session = Session(engine)
    return (session, base.classes)

def _os_value(base):
    """
    Get OS value ('windows' or 'linux') from nagios_hosts table
    base: DB schema map object
    """
    # Get value from icon column
    return case(
        [(base.nagios_hosts.icon_image_alt == 'Linux', 'linux')],
        else_='windows'
    )

def _os_filter(base, query, os):
    """
    Apply OS filter on the specified query
    base: DB schema map object
    query: SQLA query object
    os: 'windows' or 'linux'
    """
    if os == 'linux':
        value = 'Linux'
    elif os == 'windows':
        value = 'Windows Server'
    else:
        raise Exception('"{}": invalid os value'.format(os))
    return query.filter(base.nagios_hosts.icon_image_alt == value)

def _machines_filter(base, query, machine_list, how):
    """
    Apply machine filter on the specified query
    base: DB schema map object
    query: SQLA query object
    machine_list: list of machine names (case insensitive)
    how: 'include' or 'exclude'
    """
    lowered_alias = func.lower(base.nagios_hosts.alias)
    lowered_machines = [machine.lower() for machine in machine_list]
    if how == 'include':
        filters = lowered_alias.in_(lowered_machines)
    elif how == 'exclude':
        filters = lowered_alias.not_in(lowered_machines)
    else:
        Exception('"{}": invalid how value'.format(how))
    return query.filter(filters)

def get_cpu_query(
    connection_string,
    interval='hour',
    start_date=None,
    end_date=None,
    os=None,
    machines_to_include=None,
    machines_to_exclude=None,
    limit=None
    ):
    """
    Return the SQL query to fetch time series of machines cpu usage from Nagios DB according
    to input parameters, so that query returns a list of records of the form:
    (date_1, host_name_1, os, average_cpu),
    (date_1, host_name_2, os, average_cpu),
    ...
    (date_2, host_name_1, os, average_cpu),
    (date_2, host_name_2, os, average_cpu),
    ...

    where average_cpu is the average percent of CPU usage for "host_name_n" between its current and
    next record, which are separated by the specified interval, modulo actual cpu check frequency

    connection_string: DB connection string,
    eg. mysql://user:password@host/db_name
    os: filter on the specified os ('windows' or 'linux') ; unfiltered if None
    start_date: filter on data earlier than the specified datetime ; unfiltered if None
    end_date: filter on data older than the specified datetime ; unfiltered if None
    interval: 'day', 'hour', or 'minute' ; get records averaged on the specified
    interval per machine
    machines_to_include: consider only data for the specified list of machine names
    (case insensitive) ; unfiltered if None
    machines_to_exclude: ignore data for the specified list of machine names
    (case insensitive) ; unfiltered if None
    limit: maximum number of records to return ; all matching records if None
    """
    (session, base) = get_session(connection_string)

    # Truncate servicecheck start_time according to specified interval

    # year-month-day part of start_time
    servicecheck_date = func.date(base.nagios_servicechecks.start_time)
    if interval != 'day':
        # Add hour to year-month-day
        servicecheck_date = func.timestampadd(
            text('HOUR'),
            func.HOUR(base.nagios_servicechecks.start_time),
            servicecheck_date
        )
        if interval == 'minute':
            # Add minute to year-month-day-hour
            servicecheck_date = func.timestampadd(
                text('MINUTE'),
                func.minute(base.nagios_servicechecks.start_time),
                servicecheck_date
            )
        elif interval != 'hour':
            raise Exception('"{}": invalid interval value'.format(interval))

    # List of values to be retrieved per record
    values = [
        # Date
        servicecheck_date.label('date'),
        # Host name - lowercased
        func.lower(base.nagios_hosts.alias).label('machine'),
        # OS
        _os_value(base).label('os'),
        # Average CPU on the interval
        cast( # Convert from SQLA "Decimal" object to regular Float
            func.avg(
                cast( # Convert from perfdata "'cpu'=12" to int
                    func.substr(base.nagios_servicechecks.perfdata, 7),
                    Integer
                )
            ),
            Float # Raises a falsy warning, ignored line 14
        )
    ]

    query = session.query(
        *values
    ).join(
        base.nagios_services,
        base.nagios_servicechecks.service_object_id == base.nagios_services.service_object_id
    ).join(
        base.nagios_hosts,
        base.nagios_services.host_object_id == base.nagios_hosts.host_object_id
    ).filter(
        base.nagios_servicechecks.output == 'hw_usage_cpu'
    )

    # Apply date range filters
    if start_date:
        query = query.filter(servicecheck_date >= start_date)
    if end_date:
        query = query.filter(servicecheck_date <= end_date)

    # Apply OS filter
    if os:
        query = _os_filter(base, query, os)

    # Apply machines list filters
    if machines_to_include:
        query = _machines_filter(base, query, machines_to_include, 'include')
    if machines_to_exclude:
        query = _machines_filter(base, query, machines_to_exclude, 'exclude')

    # Order by servicecheck date
    query = query.order_by(asc('date'))

    query = query.group_by(
        'date',
        'machine',
        'os'
    )

    # Apply limit
    if limit:
        query = query.limit(limit)

    return query

def get_hosts_query(connection_string, os=None):
    """
    Return the SQL query to fetch all Nagios DB machines, so that query returns
    a list of records of the form:
    (host_name_1, os),
    (host_name_2, os),
    ...

    os: filter on the specified os ('windows' or 'linux') ; unfiltered if None
    """
    (session, base) = get_session(connection_string)
    query = session.query(
        # Host name - lowercased
        func.lower(base.nagios_hosts.alias).label('hostname'),
        # OS
        _os_value(base),
    )
    # Apply OS filter
    if os:
        query = _os_filter(base, query, os)
    query = query.order_by(asc('hostname'))
    return query
