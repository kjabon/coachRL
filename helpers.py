
import my_gym_wrapper

def make_coach_env(gymEnv, days, numDaysToObserve, obs):
    if gymEnv == "sim":
        import HabitEnvSim
        env = HabitEnvSim.HabitEnvSim(days, numDaysToObserve, obs)
    else:
        import HabitEnv
        import iCloud
        import HabitsSheet

        # Do a preliminary check to make sure online connections work before proceeding

        weightFile = 'WeightLog.txt'
        iCloud.DownloadFromICloud('Shortcuts', weightFile)
        weight = iCloud.GetWeight(weightFile)
        # Preprocess the value for weight.
        normalizedWeight = 1 - (160 - float(weight)) / 50
        HabitsSheet.FillInDailyWeight(normalizedWeight)
        HabitsSheet.GetObservation(future=1).flatten()

        env = HabitEnv.HabitEnv(obs)
    # env = gymWrappers.RecordVideo(env, './videos/coach_' + gymEnv + '/' + str(time()) + '/')

    # Make sure the environment obeys the dm_env.Environment interface.
    env = my_gym_wrapper.GymWrapper(env)

    return env
