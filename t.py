import easyocr

# Initialize the reader (specify the language, e.g., 'en' for English)
reader = easyocr.Reader(['en'])

# Path to the image
image_path = 'Capture1.JPG'

# Perform OCR on the image (detail=0 returns only text without bounding boxes)
result = reader.readtext(image_path, detail=0)

# Variable to store concatenated result
concatenated_text = ""

# Process each line in the result
for line in result:
    if not line.endswith('\n'):
        # If the line doesn't end with a newline, concatenate it with the next one
        concatenated_text += line + " "
    else:
        # Add the line to the concatenated text
        concatenated_text += line

# Print the concatenated text
print(concatenated_text)