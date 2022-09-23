"""
Default settings
Can be overriden by a local settings_local.py file (gitignored)
Import settings module instead of settings_default or settings_local directly
"""

# Nagios database connection string
# http://lin033dsy/nagios/
DB_CONNECTION_STRING = 'mysql+pymysql://ndoutils:nagiosadmin@lin033dsy/nagios'
