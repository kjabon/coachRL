import gym
from gym import spaces
import numpy as np
import time
import datetime
import telegram
import HabitsSheet
import iCloud
import collections


class HabitEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self,obs=[0]):
        super(HabitEnv, self).__init__()
        self.numObservations = 23
        self.numDaysToObserve = 2
        self.alpha = 0.0155
        if type(obs) == int:
            self.obsToOptimize = [obs]
        else:
            self.obsToOptimize = obs
        # Define action and observation space
        # They must be gym.spaces objects
        # See HabitEnvSim.py for more info
        self.action_space = spaces.MultiDiscrete(
            [25, 25, 25, 2, 3, 2, 5, 2, 4, 3, 12, 4, 2, 2, 2, 2, 3, 7, 5, 5, 2, 7, 10]) #164
        # Example for using image as input:
        self.observation_space = spaces.Box(low=-2, high=25, shape=(self.numObservations*self.numDaysToObserve,), dtype=np.float64)

        self.obsToActionMap = {0: 0, 1: 1, 2: 2, 3: 5, 4: 4, 5: 16, 6: 10, 7: 8, 8: 6, 9: 3, 10: 7, 11: 22, 12: 9,
                                13: 14, 14: 13, 15: 12, 16: 15, 17: 20, 18: 18, 19: 17, 20: 19, 21: 11, 22: 21}
    def step(self, action):
        #Fill in the cells in the spreadsheet corresponding to daily habits


        actionIndicesToWrite = [self.obsToActionMap[x] for x in self.obsToOptimize]
        while(HabitsSheet.FillInActionRow(action,actionsToWrite=actionIndicesToWrite) == -1):
            Wait24()
        HabitsSheet.FillInHrToMinEqns()
        #Wait 23 hours, then until 12:30 am
        Wait24()
        if 11 in self.obsToOptimize: #i.e. we are doing weight with this process
            weightFile = 'WeightLog.txt'
            targetWeight = 160  # lb
            iCloud.DownloadFromICloud('Shortcuts', weightFile)
            weight = iCloud.GetWeight(weightFile)
            # Preprocess the value for weight.
            normalizedWeight = 1 - (targetWeight - float(weight)) / 50
            HabitsSheet.FillInDailyWeight(normalizedWeight)

        #Read out the new, updated history of habit averages (observations)
        # (28 values, most recent being the day corresponding to the action just taken.)
        data = HabitsSheet.GetMonthAverageHistory(numDays=1)
        #Send a message if you should set back your sleep schedule
        sleepVal = data[-1,6]
        if sleepVal >= 0.9:
            telegram.Message("Sleep average at " + str(sleepVal) + ", decrement your wake time by 5 minutes")
        absError = abs(1.0 - data)
        todayMean = absError[-1, self.obsToOptimize].mean()
        if 0 in self.obsToOptimize:
            HabitsSheet.FillInDailyMean(1- todayMean)
            monthMean = 1-absError.mean()
            message = "Today's mean: " + str(1- todayMean) + ", monthly mean: " + str(monthMean) + ", today norm: " + str(1- todayMean - monthMean)
            telegram.Message(message)
        reward = -np.sqrt(todayMean)
        observation = HabitsSheet.GetObservation(future=self.numDaysToObserve-1)

        maskedObs = np.zeros_like(observation)
        for i in range(len(self.obsToOptimize)):
            obsToOptimize = self.obsToOptimize[i]
            maskedObs[:, obsToOptimize] = observation[:, obsToOptimize]
        maskedObs = maskedObs.flatten()
        done = False
        info = {}
        return maskedObs, reward, done, info

    def reset(self):
        #return observation  # reward, done, info can't be included
        obs = HabitsSheet.GetObservation(future = self.numDaysToObserve-1)
        maskedObs = np.zeros_like(obs)
        for i in range(len(self.obsToOptimize)):
            obsToOptimize = self.obsToOptimize[i]
            maskedObs[:,obsToOptimize] = obs[:, obsToOptimize]
        maskedObs = maskedObs.flatten()
        return maskedObs

    def render(self, mode='console'):
        pass
    def close (self):
        pass




def Wait24():
    now = datetime.datetime.now()
    waketime = now + datetime.timedelta(1) - datetime.timedelta(0,now.second,now.microsecond,0,now.minute,now.hour,0) + datetime.timedelta(0,0,0,0,30,0,0)

    message = "Current time: " + str(now) + ", waiting until next day at 12:30: " + str(waketime)
    print(message)
    time.sleep(60*60*9) #Wait 12 hours
    newNow = datetime.datetime.now()
    while newNow < waketime:
        message = "Current time: " + str(newNow) + ", waiting 30m until 12:30: " + str(waketime)
        print(message)
        time.sleep(60*30)
        newNow = datetime.datetime.now()
