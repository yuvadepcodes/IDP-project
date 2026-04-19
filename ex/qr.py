import qrcode

# Data you want to encode
data = "https://crop-sait.streamlit.app"

# Generate the QR code
img = qrcode.make(data)

# Save the image file
img.save("my_qrcode.png")

print("QR code generated successfully!")
