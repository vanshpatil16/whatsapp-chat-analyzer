import re
import pandas as pd
from datetime import datetime
def preprocessor(data):
    pattern = r"\[(\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}:\d{2} ?[APM]{2})\] ([^:]+):\s?(.+)"

    # Extract matches
    matches = re.findall(pattern, data)

    formatted_data = []
    for timestamp, user, message in matches:
        timestamp = re.sub(r' ', '', timestamp)  # Remove hidden Unicode spaces
        date_time_obj = datetime.strptime(timestamp, "%d/%m/%y, %I:%M:%S%p")

        formatted_data.append([
            date_time_obj.strftime("%d/%m/%y %H:%M:%S"),  # Full Date & Time in 24-hour format
            date_time_obj.strftime("%Y"),  # Extract Year
            date_time_obj.strftime("%B"),  # Extract Month (Full Name)
            date_time_obj.strftime("%A"),  # Extract Day (Weekday Name)
            date_time_obj.strftime("%H"),  # Extract Hour (24-hour format)
            date_time_obj.strftime("%M"),  # Extract Minute
            user.strip(),
            message.strip()
        ])

    # Convert to DataFrame
    df = pd.DataFrame(formatted_data,
                      columns=["Date & Time", "Year", "Month", "Day", "Hour", "Minute", "User", "Message"])
    return df