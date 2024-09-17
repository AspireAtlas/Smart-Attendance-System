import sqlite3

# Connect to the database
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Retrieve all rows from the 'attendance' table
cursor.execute("SELECT * FROM attendance")
rows = cursor.fetchall()

# Print the rows
for row in rows:
    print(row)

# Close the connection
conn.close()
