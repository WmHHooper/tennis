from netExamples.election import election
from netExamples.election import county_demographics
from pprint import pprint
import numpy as np
from nnet import NNet
from verbosePrint import vprint
import verbosePrint


demographics = county_demographics.get_report()
results = election.get_result()

# align demographics with results
sorted = {}
for county in demographics:
    key = county['State'] + '-' + \
          county['County'].split(' ')[0]
    inputData = [
        county['Age']['Percent 65 and Older'],
        county['Education']["Bachelor's Degree or Higher"],
        county['Education']['High School or Higher'],
        county['Ethnicities']['White Alone'],
        county['Housing']['Homeownership Rate'],
        county['Housing']['Median Value of Owner-Occupied Units'],
        county['Income']['Median Houseold Income'],
    ]
    sorted[key] = {}
    sorted[key]['inputData'] = inputData   # goal will be appended to this list

for county in results:
    key = county['Location']['State Abbreviation'] + '-' \
          + county['Location']['County'].split(' ')[0]
    trump = county['Vote Data']['Donald Trump']['Number of Votes']
    clinton = county['Vote Data']['Hillary Clinton']['Number of Votes']
    ratio = trump / (trump + clinton    # fraction voting for Trump
                     + 1E-100)          # prevent division by zero
    win = round(ratio)
    targetData = [ win ]
    if key in sorted:
        sorted[key]['targetData'] = targetData

purge = []
for county in sorted:
    if 'targetData' not in sorted[county]:
        purge += [county]

for county in purge:
    sorted.pop(county)

# pprint(sorted)

# normalize each input column into the interval [0..1]
inputRaw = np.array([sorted[k]['inputData'] for k in sorted])
iMax = inputRaw.max(0)
iMin = inputRaw.min(0)
inputData = (inputRaw - iMin) / (iMax - iMin)
inputTraining = inputData

targetRaw = np.array([sorted[k]['targetData'] for k in sorted])
tMax = targetRaw.max(0)
tMin = targetRaw.min(0)
targetData = (targetRaw - tMin) / (tMax - tMin)
targetTraining = targetData

# nn = NNet(sizes=[7, 14, 1], bias=True)
nn = NNet([[
 [ 0.83313359, -0.65108451,  0.72209966, -0.04963947, -0.95745715,
   0.75762868, -0.69654922,  0.76668433,  0.29473205, -0.74433191,
  -0.7833437 , -0.77673014,  0.05631736,  0.6367278 ],
 [-0.62270225, -0.97413056, -0.11210372,  0.51081487,  0.40079723,
  -0.89109426, -0.2681686 ,  0.41689624,  0.55243885,  0.16676638,
  -0.19907639,  0.17278213, -0.82499905, -0.3115954 ],
 [-0.50998564, -0.28771728, -0.30027269, -0.39757392,  0.77676818,
  -0.76229059,  0.14674192, -0.0766601 ,  0.49552364, -0.68928222,
  -0.04152844, -0.63884739,  0.80318165,  0.56035517],
 [ 0.94874009, -0.66176681,  0.28796773, -0.1269176 ,  0.28420912,
  -0.09841857, -0.02192694,  0.92787564,  0.14755868,  0.81810054,
  -0.73091682, -0.80275572,  0.12810028, -0.37776592],
 [-0.38769012, -0.84470618,  0.8583647 ,  0.66414179, -0.4642101 ,
   0.67484001, -0.25989566,  0.65833155, -0.11614621,  0.96436981,
  -0.5304968 ,  0.13776281,  0.58627042,  0.05139838],
 [ 0.60562616,  0.62504734,  0.25521988, -0.43570314,  0.52202195,
   0.29242763, -0.83634459, -0.31343857, -0.26510554, -0.77155353,
  -0.65211559,  0.24634917, -0.78078772, -0.00211791],
 [-0.51606073, -0.69729734,  0.48173088, -0.15028915,  0.70011867,
   0.8180747 ,  0.91873135,  0.63173191, -0.02481808, -0.4695544 ,
  -0.29850299, -0.1578929 , -0.43877302,  0.06210981],
 [-0.03807058, -0.8733925 ,  0.03089031,  0.23816943,  0.81860479,
  -0.27525149,  0.27003075,  0.61380241,  0.46363094,  0.70720887,
  -0.85315039,  0.38304242,  0.67589639, -0.8194846 ]],
  [
[ 1.29233569],
 [ 0.34499149],
 [ 0.74635881],
 [-0.08937701],
 [ 0.58916744],
 [ 1.55539952],
 [ 0.26657558],
 [-0.93090432],
 [-0.58472614],
 [ 1.19865276],
 [ 0.58429752],
 [-1.67456453],
 [-0.40517135],
 [-0.49674854],
 [-0.48503573]
  ]], bias=True)
            # ]])
nn.setActivations(['tanh', 'sigmoid'])
nn.setAlpha(0.01)
nn.setVerbose([])

nn.checkup(inputData, targetData)

verbosePrint.vIteration = -1
verbosePrint.stage = ''

cycles = 10
report = cycles/10

for iteration in range(cycles + 1):
    vprint(iteration, '~~~~~~~~~~~ Iteration %d ~~~~~~~~~~~' % iteration)
    combinedError = 0
    for row_index in range(len(targetTraining)):
        datain = inputTraining[row_index:row_index + 1]
        goal_prediction = targetTraining[row_index:row_index + 1]
        prediction = nn.fire(datain)
        vprint(iteration, nn)

        error = (goal_prediction - prediction) ** 2
        combinedError += error

        nn.learn(datain, goal_prediction)

    if iteration % report == 0:
        print('Iteration: %d Error: %s' % (iteration, str(combinedError)))
    vprint(iteration, '')
    vprint(iteration, '~~~~~~~~~~~~~~~ End ~~~~~~~~~~~~~~~~')
    vprint(iteration, nn, quit=True)

print()
nn.checkup(inputData, targetData)
