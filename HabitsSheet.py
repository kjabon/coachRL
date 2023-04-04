
import datetime
from datetime import date, timedelta
import numpy as np
import copy
from googleapiclient.discovery import build #pip install google-api-python-client
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import os
import telegram
import random
import string
import iCloud
import time

#You need to set up api access on your google account, and get the id corresponding to your created habits spreadsheet.
#Google has good documentation for this, it's not too hard. Good luck!
defaultID = 'YourIDHere'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
obsToActionMap = {0: 0, 1: 1, 2: 2, 3: 5, 4: 4, 5: 16, 6: 10, 7: 8, 8: 6, 9: 3, 10: 7, 11: 22, 12: 9,
                                13: 14, 14: 13, 15: 12, 16: 15, 17: 20, 18: 18, 19: 17, 20: 19, 21: 11, 22: 21}
#get a handle to the google api
def GetService():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('sheets', 'v4', credentials=creds)

#read one or more rows of EMAs from the google sheet, into the future for your observation
def GetObservation(spreadsheet_id = defaultID, sheetRange = "Habits", future = 0):
    # First, update sheet with most recent weight. Then get the whole dataset as an obs.
    service = GetService()
    #You need to pick the appropriate column numbers to read for your personal usage.
    avgCols = [24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]#24 to 46
    history = np.zeros((future+1, len(avgCols)))  # 4 weeks

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=sheetRange).execute()
    except HttpError as error:
        print(f"An error occurred: {error}")
        return error

    rows = result.get('values', [])
    rowIndex = 0
    rowNum = 0
    for row in rows:
        rowIndex += 1
        if rowIndex == 1:
            continue
        currDate = RowToDate(row)
        if currDate > date.today() + timedelta(days = future) or currDate < date.today():
            continue
        colNum = 0
        for colIndex in avgCols:
            history[rowNum, colNum] = row[colIndex]
            colNum += 1
        rowNum += 1
    return history

#read one or more rows of EMAs from the google sheet, into the PAST for your observation
def GetMonthAverageHistory(spreadsheet_id = defaultID, sheetRange = "Habits", numDays = 28):
    #First, update sheet with most recent weight. Then get the whole dataset as an obs.
    service = GetService()
    # You need to pick the appropriate column numbers to read for your personal usage.
    avgCols = [24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]
    history = np.zeros((numDays,len(avgCols))) #4 weeks

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=sheetRange).execute()
    except HttpError as error:
        print(f"An error occurred: {error}")
        return error

    rows = result.get('values', [])
    rowIndex = 0
    rowNum = 0
    for row in rows:
        rowIndex += 1
        if rowIndex == 1:
            continue
        currDate = RowToDate(row)
        if currDate > date.today() or currDate <= date.today() - timedelta(days=numDays):
            continue
        colNum = 0
        for colIndex in avgCols:
            history[rowNum, colNum] = row[colIndex]
            colNum += 1
        rowNum += 1
    return history

#I like to track the mean performance in the sheet, this just picks out the right column for that
def FillInDailyMean(todayMean):
    colDesc = "BB"
    WriteTodayRow(colDesc, [todayMean])

#Same as above, but for today's weight
def FillInDailyWeight(weight):
    colDesc = "AJ"
    WriteTodayRow(colDesc,[weight])

#Note values must be in python list form, even if only one value
def WriteTodayRow(colDesc, values):
    WriteTodayRowOffset(0,colDesc,values)

def GetTodayRowNum(service=None, spreadsheet_id = defaultID, sheetRange = "Habits"):
    if service == None:
        service = GetService()
    try:
        # Get the row num
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=sheetRange).execute()

    except HttpError as error:
        print(f"An error occurred: {error}")
        return error

    rows = result.get('values', [])
    rowIndex = 0
    for row in rows:
        # Print columns A and E, which correspond to indices 0 and 4.
        rowIndex += 1
        if rowIndex == 1:
            continue
        currDate = RowToDate(row)
        if currDate == date.today():
            break
    return rowIndex

def RowToDate(row):
    currDate=row[0]
    currDate = datetime.datetime.strptime(currDate, "%b %d, %Y").date()
    return currDate

#This function will massage the action output by the NN before writing to the Google Sheet.
#See HabitEnvSim function for details
def FillInActionRow(action, actionsToWrite=-1):
    #Preprocess actions
    a = copy.deepcopy(action).astype(np.float32)
    for i in range(3):
        a[i] /= 6
    observations = GetObservation().flatten()
    workObs = observations[0:3]
    meanObs = np.mean(workObs)
    y = -1.33*meanObs*4 + 9.33 #In the interest of keeping a steady work schedule from day to day, I hard coded work hours.
    a[0] = y*120/240
    a[1] = y*70/240
    a[2] = y*50/240


    a[4] = a[4] - 1
    a[16] = a[16] - 1

    a[9] = a[9]/2
    a[10] = a[10]/10
    #Rest
    if round(a[11]) == 0 or round(a[11]) == 1 or round(a[11]) == 2:
        a[11] = 0
    else:
        a[11] = 1


    # The sum of music, MEH, and blog should not be greater than 6; decrement
    sumExtraCurricular = a[17] + a[18] + a[19] + a[21]
    indicesToDec = [17, 18, 19, 21]
    while sumExtraCurricular > 4:
        random_index = random.randint(0, len(indicesToDec) - 1)
        indexToDec = indicesToDec[random_index]
        if a[indexToDec] > 0:
            a[indexToDec] -= 1
        sumExtraCurricular = a[17] + a[18] + a[19] + a[21]

    # No more than one exercise per day, except for stretching. Decrement ath, cardio in that order
    sumExercise = a[12] + a[13] + a[14] + a[15]
    indicesToDec = [12,13,14,15]
    while sumExercise > 1:
        random_index = random.randint(0,len(indicesToDec)-1)
        indexToDec = indicesToDec[random_index]
        if a[indexToDec] > 0:
            a[indexToDec] -= 1
        sumExercise = a[12] + a[13] + a[14] + a[15]


    weightFile = 'WeightLog.txt'
    iCloud.DownloadFromICloud('Shortcuts', weightFile)
    weight = float(iCloud.GetWeight(weightFile))
    if weight > 165:
        a[22] = (1600 + a[22] * 100 + min(2500, max(8820 - 124 / 3 * weight, 1380))) / 2
    else:
        a[22] = 1600 + a[22] * 100

    # Cheat; i.e. if you know something has to always be 1 to reach 1, just set it as such and skip learning this.
    a[4] = 1
    a[5] = 1
    a[7] = 1
    a[10] = 1

    # Get x and b obs; these are keystone habits that I gauge everything else around
    x = observations[4]
    y = observations[5]
    if y < 0.8*x:
        x[16] = 1

    # Now, for most actions, if the observation is > y, set it to 0: take a breather, you're doing well!
    for obsIndex in [6]+list(range(12,23)):
        if observations[obsIndex] > y:
            a[obsToActionMap[obsIndex]] = 0

    # Except for brush and skin care, those are 1
    for obsIndex in [7,8]:
        if observations[obsIndex] > y:
            a[obsToActionMap[obsIndex]] = 1

    letters = GetActionLettersToWrite(actionsToWrite)
    for i, letter in enumerate(letters):
        if not (i==0 or i == 1 or i ==2) and TodayActionIsFilled(letter):
            print("Action for today was already entered.")
            telegram.Message("Action for today was already entered.")
            return -1
        WriteTodayRow(letter, [float(a[actionsToWrite[i]])])

    return 0

# Gets the action columns for my particular configuration of the habits sheet on google sheets.
# (There is almost certainly a more elegant way to handle this)
def GetActionLettersToWrite(actionIndicesToWrite):
    if actionIndicesToWrite == -1 or len(actionIndicesToWrite) == 23:
        return "B:X"
    uppercaseLetters = list(string.ascii_uppercase)
    actionRowLetters = uppercaseLetters[1:24 + 1]  # B:X
    columnsToWrite = [actionRowLetters[x] for x in actionIndicesToWrite]
    return columnsToWrite

# Determines if the cells are already written to; in the case of debugging and restarting the program, we use this to
# not overwrite a written action row in google sheets.
def TodayActionIsFilled(letter = 'B:X', spreadsheet_id = defaultID):

    service = GetService()
    rowIndex = GetTodayRowNum(service)
    colDesc = letter
    colDescs = colDesc.split(":")
    if len(colDescs) == 2:
        entryRange = colDescs[0] + str(rowIndex) + ":" + colDescs[1] + str(rowIndex)
    else:
        entryRange = colDesc + str(rowIndex)
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=entryRange).execute()
        values = result.get('values', [])
        try:
            count = sum(map(lambda x : x != '', values[0]))
        except: count = 0


    except HttpError as error:
        print(f"An error occurred: {error}")
        return error
    return count == 23 or count == 1


# Writes to today's row for dayOffset = 0, yesterday for dayOffset = -1, and so on
def WriteTodayRowOffset(dayOffset, colDesc, values, spreadsheet_id=defaultID):
    # Note values must be in python list form
    service = GetService()
    time.sleep(8)
    rowIndex = GetTodayRowNum(service) + dayOffset
    colDescs = colDesc.split(":")
    if len(colDescs) == 2:
        entryRange = colDescs[0] + str(rowIndex) + ":" + colDescs[1] + str(rowIndex)
    else:
        entryRange = colDesc + str(rowIndex)
    # Fill in row with actions
    body = {
        'values': [values]
    }
    try:
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range=entryRange,
            valueInputOption="USER_ENTERED", body=body).execute()

    except HttpError as error:
        print(f"An error occurred: {error}")
        return error

    return result

# Work is entered in units of hours. For convenience, I also enter this as minutes a couple rows below.
def FillInHrToMinEqns(spreadsheet_id = defaultID, sheetRange = "Habits"):
    colDesc = "B:D"
    rowIndex = GetTodayRowNum()
    values = ["=B"+str(rowIndex)+"*60","=C"+str(rowIndex)+"*60","=D"+str(rowIndex)+"*60"]
    WriteTodayRowOffset(1,colDesc,[0,0,0])
    WriteTodayRowOffset(2,colDesc,values)

