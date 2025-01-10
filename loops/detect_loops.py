from os import listdir, scandir
from subprocess import run
#from sys import argv

"""
This script was used in the Lattice Multigec 2024 participation to detect loops in essay generation.

Mathieu Dehouck
11/2024
"""

#
# This is not clean deployment code, this was used in a shared-task as a diagnostic tool.
#


def repeat(x, k):
    """
    looks for repetitions of length k in a string x.
    Note : the hard coded parameters could be search more formally.
    """
    for i in range(len(x), -1, -10):
        reps = []
        s = x[i:i+k] # this is the section we're looking for
        for j in range(len(x)):
            if x[j:j+k] == s:
                reps.append(j) # all the positions at which we find s

        if len(reps) > 20: # if we found it more than 20 times
            print('Found :', '"'+s+'"', len(reps),'times at', reps)

            delta = [p-r for (p,r) in zip(reps[1:], reps)] # compute intervals
            repps = set(delta) # number of different intervals
            print('Intervals :', delta, 'making', len(repps), 'periods', repps)
            if len(repps) < 4: # if we have less than a given number of different interval we likely have cycles
                if '.' in x[reps[0]:reps[1]]: # look for sign of sentences
                    p = x.find('.', reps[0])
                    #print('YES', x.find('.', reps[0]), x[:p])
                    return 'Many', x[:x.find('.', reps[0])+1]
                return 'Many', s.join(x.split(s)[:2]) # we haven't found a ., let the participant check that sentence by hand
            return True, x # seems that we have lots of repetitions, but the cycle may be more complicated than expected

    return False, x # no cycle for the given hardcoded parameters


mlg = 'path' # the path to the multigec folder
for fl in sorted(scandir(mlg), key=lambda x: x.name): 
    
    if 'test.md' in fl.name and '~' not in fl.name: # check each hypo-test.md one by one
        good = True
        count = [0,0] # keep track of the total number of essays and of essays with cycles

        fin = open(mlg+fl.name)
        fout = open(mlg+'cleaned/'+fl.name, 'w') # the cleaned folder is used to avoid erasing original outputs

        for l in fin: # line by line
            l = l.strip()

            if l == '':
                continue

            if l.startswith('###'): # new essay
                count[0] += 1
                idx = l
                print(l, file=fout)
                continue


            # check for repetition            
            r, x = repeat(l, 15)

            if r == 'Many':
                print(x, file=fout)
                good = False
                count[1] += 1
                print(fl.name, idx, "#####################") # dirty but efficient to travel in the logs
                continue

            elif r == True:
                good = False
                count[1] += 1
                print(fl.name, idx, l[1:]) # in case of potential loops with weird cycle length, we output it to the log for the user

            
            if '$$$' in l: # in case we have not already removed all the end of line markers
                l = l.split('$$$')
                if len(l) == 2 and l[1] == '':
                    print(l[0], file=fout)
                    continue
                #print(fl.name, idx, l[1:])

            else:
                print(l, file=fout)
                

        if good:
            print('No problem found in :', fl.name)
        else:
            print('LOOPS FOUND IN :', fl.name, count)

