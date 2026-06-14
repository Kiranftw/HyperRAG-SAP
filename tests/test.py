import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
from gen_ai_hub.proxy.native.google_genai.clients import Client
from PIL import Image
# from langchain.parse import SimpleJsonOutputParser()

def main():
    client = Client()
    image_path = "/home/kiranftw/HyperRAG-SAP/tests/cancelled_cheque.jpeg"
    try:
        image = Image.open(image_path)
        print(image)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    prompt = "Extract all data from this cheque and provide a clear AI summary of the details."
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            # deployment_id = "ddf12583de56f792",
            contents=[
                prompt,
                image
            ]
        )
        print("Success! AI Summary of the cheque:\n")
        print(response.text)
    except Exception as e:
        print("Error:", type(e), e)

if __name__ == "__main__":
    main()



# Here's a summary of the cheque details extracted from the image:

# **Cheque Issuing Bank:** YES BANK Ltd.
# **Branch Address:** Mayank Towers, Survey No. 31 (Old), 31/2 (New), Raj Bhavan Road, Somajiguda, Hyderabad - 500082
# **IFS Code:** YESB0000006
# **Account Type:** CURRENT
# **Account Number:** 000681300007561
# **Payable:** At Par At All Branches of YES BANK Ltd.
# **Cheque Number:** This appears to be encoded in the MICR line as "130053 500532001 040055 29".
# **Payee:** The handwritten portion "or Bearer" or "या धारक को" indicates the payment can be made to the bearer of the cheque. The stamp at the bottom indicates the cheque is "For NCL BUILDTEK LTD".
# **Amount:** The amount in words is "Rupees रुपये" and in figures, it appears to be ₹ 040055.
# **Date:** The date is to be filled in the "DDMMYYYY" boxes.
# **Validity:** The cheque is valid for three months from the date of issue.
# **Other Information:**
# * "New Account" is stamped on the cheque.
# * "Authorised Signatories Please sign above" is printed.
# * The cheque is printed by "UTILITY FORMS PVT. LTD. / CTS - 2010".

# **AI Summary:**

# This is a cheque issued by YES BANK Ltd. from their Somajiguda, Hyderabad branch. It is a CURRENT account cheque with the account number 000681300007561. The cheque is payable to the bearer for an amount of ₹ 040055 (Four Lakh Five Thousand and Fifty-Five Paisa, assuming the leading "0" before 40055 is part of the numerical representation). The date needs to be filled in the provided fields, and the cheque is valid for three months from the date of issue. The cheque is intended for NCL BUILDTEK LTD and requires authorized signatures. The cheque number is encoded in the MICR line.