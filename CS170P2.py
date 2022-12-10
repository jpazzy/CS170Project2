import numpy as np
import math

def leaveOneOutCrossValidation(_data, current_set, feature_to_add):
    # Append new feature to add to current set
    keptColumns = [0] + current_set +[feature_to_add] 
    # Extract only the columns which we are concered about
    # plus the label column
    numberCorrectlyClassified = 0
    data = _data[:,keptColumns]
    for i in range(len(data)):
        objectToClassify = data[i][1:]
        
        labelObjectToClassify = data[i][0]
        nearestNeighborDistance = np.inf
        nearestNeighborLocation = np.inf
        nearestNeighborLabel = 0
        
        for j in range(len(data)):
            if i != j:              
                distance = np.dot((objectToClassify - data[j][1:]),(objectToClassify - data[j][1:]))
               
                if distance < nearestNeighborDistance:
                    nearestNeighborDistance = distance
                    nearestNeighborLocation = j
                    nearestNeighborLabel = data[nearestNeighborLocation][0]
                    
        if labelObjectToClassify == nearestNeighborLabel:
            numberCorrectlyClassified += 1
            
    return numberCorrectlyClassified / len(data)
                           
def leaveOneOutCrossValidationBackwards(_data, _current_set, feature_to_remove):
    # Remove new feature from list
    keptColumns = _current_set.copy();
    keptColumns.remove(feature_to_remove)
    # Extract only the columns which we are concered about
    # plus the label column
    numberCorrectlyClassified = 0
    data = _data[:,keptColumns]
    for i in range(len(data)):
        objectToClassify = data[i][1:]
        labelObjectToClassify = data[i][0]
        nearestNeighborDistance = np.inf
        nearestNeighborLocation = np.inf
        nearestNeighborLabel = 0
        
        for j in range(len(data)):
            if i != j:              

                distance = np.dot((objectToClassify - data[j][1:]),(objectToClassify - data[j][1:]))
               
                if distance < nearestNeighborDistance:
                    nearestNeighborDistance = distance
                    nearestNeighborLocation = j
                    nearestNeighborLabel = data[nearestNeighborLocation][0]
                    
        if labelObjectToClassify == nearestNeighborLabel:
            numberCorrectlyClassified += 1
            
    return numberCorrectlyClassified / len(data)
                           
def featureSearch(data):
    currentSet = []
    accuracySet = []
    for i in range(len(data[0])-1):
        print("On the " + str(i+1) + "th level of the search tree!")
        
        featureToAdd = 0
        bestAccuracySoFar = 0
        for j in range(len(data[0])-1):
            if (j+1) not in currentSet:
                print("---Considering adding the " + str(j+1) + " feature")
                accuracy = leaveOneOutCrossValidation(data, currentSet, j+1)
                if accuracy > bestAccuracySoFar:
                    bestAccuracySoFar = accuracy
                    featureToAdd = j
        print("Added feature " + str(featureToAdd+1) +" to the list!")
        currentSet.append(featureToAdd+1)
        accuracySet.append([bestAccuracySoFar,currentSet.copy()])
        
    return (accuracySet)

def featureSearchBackwards(data):
    currentSet = list(range(0,len(data[0])))
    accuracySet = []
    for i in range(len(data[0])-1):
        print("On the " + str(i+1) + "th level of the search tree!")
        
        featureToAdd = 0
        bestAccuracySoFar = 0
        for j in range(len(data[0])-1):
            if (j+1) in currentSet:
                print("---Considering Removing the " + str(j+1) + " feature")
                accuracy = leaveOneOutCrossValidationBackwards(data, currentSet, j+1)
                if accuracy > bestAccuracySoFar:
                    bestAccuracySoFar = accuracy
                    featureToAdd = j
        print("Removed feature " + str(featureToAdd+1) +" from the list!")
        currentSet.remove(featureToAdd+1)
        copy = currentSet.copy()
        copy.remove(0)
        accuracySet.append([bestAccuracySoFar,copy])
        
    return (accuracySet)

def main():
    print("Welcome to Justin's Feature Selection Algorithm.")
    print("Type in the name of the file to test : ")
    fileName = input()
    print("Type the number of the algorithm you want to run.")
    print("     1) Forward Selection")
    print("     2) Backward Elimination")
    choice = input()
    
    featureSet = []
    data = np.loadtxt(fileName)
    if choice == '1':
        featureSet = featureSearch(data)
    else:
        featureSet = featureSearchBackwards(data)
    maxSet = max(featureSet)
    print("\nThe best feature set found was " + str(maxSet[1]))
    print("With an accuracy of " + str(maxSet[0]))
if __name__ == "__main__":
    main()
