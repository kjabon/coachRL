import sys
from pyicloud import PyiCloudService #pip install pyicloud
from shutil import copyfileobj
import datetime


def GetAPI():
    #Note there are more secure ways to do this, and text passwords are bad practice; add them at your discretion
    api = PyiCloudService('youricloudusername', 'youricloudpassword')
    Do2FA(api)
    return api

# Downloads arbitrary files from icloud, this is used to get weight data saved by iphone by "Shortcuts" app
def DownloadFromICloud(foldername, filename):
    api = GetAPI()
    drive_file = api.drive[foldername][filename]

    with drive_file.open(stream=True) as response:
        with open(drive_file.name, 'wb') as file_out:
            copyfileobj(response.raw, file_out)

#Assumes you've downloaded a txt file from icloud containing daily weight measurements; see WeightLog.txt for example
def GetWeight(filename):
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    for line in Lines:
        lineItems = line.strip().split(' ')
        dateStr = lineItems[0]
        date = datetime.datetime.strptime(dateStr, '%Y-%m-%d').date()
        if date == datetime.datetime.today().date():
            return lineItems[1]
    lineItems = Lines[-1].strip().split(' ')
    return lineItems[1]

def Do2FA(api):
    if api.requires_2fa:
        print("Two-factor authentication required.")
        code = input("Enter the code you received of one of your approved devices: ")
        result = api.validate_2fa_code(code)
        print("Code validation result: %s" % result)

        if not result:
            print("Failed to verify security code")
            sys.exit(1)

        if not api.is_trusted_session:
            print("Session is not trusted. Requesting trust...")
            result = api.trust_session()
            print("Session trust result %s" % result)

            if not result:
                print("Failed to request trust. You will likely be prompted for the code again in the coming weeks")
    elif api.requires_2sa:
        import click
        print("Two-step authentication required. Your trusted devices are:")

        devices = api.trusted_devices
        for i, device in enumerate(devices):
            print(
                "  %s: %s" % (i, device.get('deviceName',
                                            "SMS to %s" % device.get('phoneNumber')))
            )

        device = click.prompt('Which device would you like to use?', default=0)
        device = devices[device]
        if not api.send_verification_code(device):
            print("Failed to send verification code")
            sys.exit(1)

        code = click.prompt('Please enter validation code')
        if not api.validate_verification_code(device, code):
            print("Failed to verify verification code")
            sys.exit(1)