import random
import gym
from gym import spaces
import numpy as np
import collections
import copy


# This class is an env in which to perform simulated training of the Coach in parallel
# It ideally simulates real life behavior, though we can really only simulate an ideal scenario where all suggestions are followed.
class HabitEnvSim(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, numDays=1, numDaysToObserve=1, obsToOptimize=[0]):
        super(HabitEnvSim, self).__init__()

        self.numObservations = 23 #Sets the size of the observation space, equal to the number of habits

        if type(obsToOptimize) == int:
            # Sets the observations to look at during simulation. If some are skipped, they are set to 0 and provide no learning signal
            self.obsToOptimize = [obsToOptimize]
        else:
            self.obsToOptimize = obsToOptimize

        self.timeStep = 0
        #For EMA
        self.alpha = 0.0155
        self.weight = 160 + random.uniform(-10, 25) #Randomly initialize weight
        # Number of days
        self.timeThreshold = 1600 #This just sets the max episode length
        self.numDays = numDays  # How many days to keep hanging around for observing? (Large vals for debugging)

        self.numDaysToObserve = numDaysToObserve #How many days to use for the EMA observation
        # Define action and observation space
        # They must be gym.spaces objects

        #This defines the amount of possible discrete actions for each habit; your list may be much shorter
        self.action_space = spaces.MultiDiscrete(
            [25, 25, 25, 2, 3, 2, 5, 2, 4, 3, 12, 4, 2, 2, 2, 2, 3, 7, 5, 5, 2, 7, 10], dtype=np.int32)

        # In case, in your google sheet, your action columns are not sorted in the same order as their corresponding observation (EMA) columns
        # We keep a dict of which observations correspond to which actions
        self.obsToActionMap = {0: 0, 1: 1, 2: 2, 3: 5, 4: 4, 5: 16, 6: 10, 7: 8, 8: 6, 9: 3, 10: 7, 11: 22, 12: 9,
                               13: 14, 14: 13, 15: 12, 16: 15, 17: 20, 18: 18, 19: 17, 20: 19, 21: 11, 22: 21}

        # Converts the action into the observed, normalized value. E.g., if I want to perform a habit 6 days a week, I would multiply the action by 7/6
        # In this way, if the action is "1" 6 days a week and "0", this will give me a normalized EMA of 1. (6*(7/6) + 0*(7/6) = 7)
        #One for each action/habit
        self.actionMultipliers = np.array([60 / 120, 60 / 70, 60 / 50, 7 / 6,
                                           1, 1, 1 / 2, 7 / 6,
                                           7 / 6 / 3, 7 / 1, 1, 7, 7 / 2/0.8, 7/0.8, 7 / 2/0.8,
                                           7 / 2/0.8, 1, 7 / 4, 7 / 2, 7 / 5, 7 / 2, 1/2, -1])


        obsSpaceSize = self.numDaysToObserve * self.numObservations
        self.observation_space = spaces.Box(low=-2, high=10, shape=(obsSpaceSize,), dtype=np.float64)

        self.actionHistory = collections.deque(maxlen=self.numDays)
        self.history = collections.deque(maxlen=self.numDays)
        #Initialize the history
        for _ in range(self.numDays):
            currAction = self.action_space.sample()
            self.FillInActionRow(currAction, pop=False)
            self.UpdateMonthAverage(pop=False)

    def step(self, action):

        # Fill in the cells in the spreadsheet corresponding to daily habits
        self.FillInActionRow(action)
        self.UpdateMonthAverage()

        # Read out the new, updated history of habit averages (observations)
        # (28 values, most recent being the day corresponding to the action just taken.)
        data = self.GetMonthAverageHistory()
        absError = abs(1.0 - data)
        todayErr = absError[-1, self.obsToOptimize].mean()
        reward = -todayErr / self.timeThreshold

        observation = data[-1, :].reshape((1, self.numObservations))
        if self.numDaysToObserve > 1: #Potentially expand the observation to include future habit decay; this is defunct
            obsTomorrow = self.UpdateMonthAverage(future=self.numDaysToObserve - 1)
            observation = np.concatenate((observation, obsTomorrow))
        done = False

        if self.timeStep >= self.timeThreshold:
            done = True
        else:
            self.timeStep += 1
        info = {}
        return observation.flatten(), reward, done, info

    def reset(self):
        # Randomly initialize a full queue's worth of history of actions and the final weight
        self.timeStep = 0
        for _ in range(self.numDays):
            currAction = self.action_space.sample()
            self.FillInActionRow(currAction)
            self.UpdateMonthAverage()

        #Initialize weight randomly
        self.weight = 160 + random.uniform(-10, 25)

        #Return the most recent day for the first observation, optionally adding future habit decay (this is defunct)
        obs = self.GetMonthAverageHistory()[-1, :].reshape((1, self.numObservations))
        if self.numDaysToObserve > 1:
            obsTomorrow = self.UpdateMonthAverage(future=self.numDaysToObserve - 1)
            obs = np.concatenate((obs, obsTomorrow)).flatten()
        return obs.flatten()

    def render(self, mode='console'):
        pass


    def close(self):
        pass

    def FillInActionRow(self, action, pop=True):
        # Preprocess actions. Action vector is an int vector, with ith action = an int from 0 to self.action_space[i] -1
        # Sometimes we would like to convert these ints into a more meaningful habit-specific representation,
        # which we do below before saving them to the actual habit "actions" for the day
        a = copy.deepcopy(action).astype(np.float32)

        # E.g.: each int represents a[i] * 10 minutes. E.g., a[i] = 2 (<- the int output!) represents 20 minutes
        # So we divide by 6 here, and convert to hours. (2/6 = 0.333 hours, same as 20 minutes)

        #The details of this section are not important, and should be replaced by your own code.
        #They merely serve as inspiration/ examples.
        for i in range(3):
            a[i] = a[i] / 6

        a[4] = a[4] - 1
        a[16] = a[16] - 1

        a[9] = a[9] / 2
        a[10] = a[10] / 10
        # Rest
        if round(a[11]) == 0 or round(a[11]) == 1 or round(a[11]) == 2:
            a[11] = 0
        else:
            a[11] = 1

        if len(self.history) > 0:
            observations = self.history[-1]
            workAvg = np.mean(observations[0:3])
            hrsToWork = 11 - (6 + 2 / 3) * workAvg
            hrsToWork = max(min(hrsToWork,5+1/3),2)

        else:
            hrsToWork = 5 + 1 / 3
        workSurplusRatio = (a[0] + a[1] + a[2]) / hrsToWork
        if workSurplusRatio > 1:
            for i in range(3):
                a[i] /= workSurplusRatio + 1e-3  #
        # The sum of music, MEH, and blog should not be greater than 6; decrement
        sumExtraCurricular = a[17] + a[18] + a[19]
        indicesToDec = [17, 18, 19]
        while sumExtraCurricular > 4:
            random_index = random.randint(0, len(indicesToDec) - 1)
            indexToDec = indicesToDec[random_index]
            if a[indexToDec] > 0:
                a[indexToDec] -= 1
            sumExtraCurricular = a[17] + a[18] + a[19]

        # No more than one exercise per day, except for stretching. Decrement randomly until this is true
        sumExercise = a[12] + a[13] + a[14] + a[15]
        indicesToDec = [12, 13, 14, 15]
        while sumExercise > 1:
            random_index = random.randint(0, len(indicesToDec) - 1)
            indexToDec = indicesToDec[random_index]
            if a[indexToDec] > 0:
                a[indexToDec] -= 1
            sumExercise = a[12] + a[13] + a[14] + a[15]

        # Cheat: if you know you always want to set a habit to 1, just do so to speed up learning
        a[4] = 1
        a[5] = 1

        if pop:
            self.actionHistory.popleft()  # toss old days once queue is full
        self.actionHistory.append(a)

    def UpdateMonthAverage(self, pop=True, future=0):

        # Just need yesterday's history, and today's actions
        actions = np.asarray(self.actionHistory)[-1, :]
        if len(self.history) > 0:
            history = np.asarray(self.history)[-1, :]
        else:
            history = np.zeros(self.numObservations)
        # Now, don't bother with this function. Just calculate it!
        # Today val = yesterday(1-a) + today(a)
        scaledToday = np.zeros(self.numObservations)
        temp = self.alpha * actions * self.actionMultipliers
        temp = np.asarray([temp[self.obsToActionMap[i]] for i in self.obsToOptimize])
        for i in range(len(self.obsToOptimize)):
            obs = self.obsToOptimize[i]
            scaledToday[obs] = temp[i]
        scaledTomorrow = (1 - self.alpha) * history
        h = scaledToday + scaledTomorrow

        if future > 0:
            hs = np.zeros((future, self.numObservations))
            currHistory = history
            for i in range(future):
                hs[i, :] = (1 - self.alpha) * currHistory
                currHistory = hs[i, :]
            return hs

        h[11] = self.GetDailyWeight()

        if pop:
            self.history.popleft()
        self.history.append(h)


    def GetMonthAverageHistory(self):
        history = np.concatenate(self.history).reshape(
            (self.numDays, self.numObservations))  # might need to stick this into a numpy ndarray

        return history

    #Simulate a weight update based on exercise and calories
    def GetDailyWeight(self):
        if len(self.actionHistory) < self.numDays or len(
                self.history) < self.numDays or self.numDays < 2 or 11 not in self.obsToOptimize:
            return 0.0
        baseline = 2000
        targetWeight = 160.0
        # Get diet (y/n)
        actions = self.actionHistory[-2]  # 7 diet 22 calories


        if self.weight > 165:
            calories = (1600 + actions[22] * 100 + min(2500, max(8820 - 124 / 3 * self.weight, 1380))) / 2
        else:
            calories = 1600 + actions[22] * 100

        # Get yesterday's exercise (not stretch)
        exercise = actions[13] + actions[14] + actions[15]
        # Get yesterday's weight
        calories = calories - exercise * 250
        delta = (calories - baseline) / 3500
        self.weight += delta
        normalizedWeight = 1 - (targetWeight - self.weight) / 25
        return normalizedWeight  # Weight change will need to be calculated based on calorie intake and exercise

