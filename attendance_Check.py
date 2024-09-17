import sqlite3

# Connect to the database
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Check if the 'attendance' table exists in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attendance'")
table_exists = cursor.fetchone()

# Close the connection
conn.close()

# Print the result
if table_exists:
    print("The 'attendance' table exists in the database.")
else:
    print("The 'attendance' table does not exist in the database.")
