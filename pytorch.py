array = [50 ,68 , 86 ]
for i in range(len(array)) :
    var = 1
    if i == len(array) - 1 :
        var = 0
        
    if ( array[i] - array[i + var ] >= 20 or array[i] - array[i + var ] >= - 20 ) and array[i] % 2 == 0 :
        if var == 0 :
            print("found cat")
    else :
        print("not a hat")
        break
# basic idea how pytorch or deep learnings finds the pattern
