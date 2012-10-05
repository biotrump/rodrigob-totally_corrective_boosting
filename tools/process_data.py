# Script for data management
# 1. Checks for the existence of all data sets
# 2. If data does not exist, it will download it
# 3. Data sets that have already been split into
#    train and test files are concatenated
# 4. Data sets are divided into train, test, and validation
#    sets with the following ratio:
#    60% train, 20% test, 20% validation

import os
import shutil
import string
import numpy as np


def download(url):
	"""Copy the contents of a file from a given URL
	to a local file.
	"""
	import urllib
	webFile = urllib.urlopen(url)
	localFile = open(url.split('/')[-1], 'w')
	localFile.write(webFile.read())
	webFile.close()
	localFile.close()

def data_split(ifile,ofile):
    """ 
    The function will read in a dataset from ifile, 
    split it so that 60% of the data is in the training set,
    20% is in the test set, and 20% is in the validation set.
    It will then write these data to the files:
              ofile.train, ofile.test, ofile.valid
    Parameters
    ----------
    ifile: string
              The name of the file where the data resides
    ofile: string
              The root name of the output file

    """

    f = open(ifile,'r')
    a = f.readlines()
    f.close()

    m = len(a)
 
    np.random.seed(4748590902)
    perm = np.random.permutation(len(a))
    
    train = perm[0:int(.6*m)]
    valid = perm[int(0.6*m):int(0.8*m)]
    test = perm[int(0.8*m)::]


    trainfile = ofile+".train"
    if not(os.path.isfile(trainfile)):
        print "\tcreating %s"%(trainfile)
        ftr = open(trainfile, 'w')
        for i in train:
            ftr.write(a[i])
        ftr.close()
    else:
        print "\t%s exists"%(trainfile)

    testfile = ofile+".test"
    if not(os.path.isfile(testfile)):
        print "\tcreating %s"%(testfile)
        fte = open(testfile,'w')
        for i in test:
            fte.write(a[i])
        fte.close()
    else:
        print "\t%s exists"%(testfile)

    validfile = ofile+".valid"
    if not(os.path.isfile(validfile)):
        print "\tcreating %s"%(validfile)
        fv = open(validfile,'w')
        for i in valid:
            fv.write(a[i])
        fv.close()
    else:
        print "\t%s exists"%(validfile)

#-----------------------------------------
# define urls for data sets

datasets = {}
datasets['a9a'] = \
    'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a'
datasets['a9a.t'] = \
    'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t'
# datasets['rcv1_train.binary'] = \
#     'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
# datasets['rcv1_test.binary'] = \
#     'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2'
datasets['real-sim'] = \
    'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2'
datasets['news20.binary'] = \
    'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2'
datasets['german.numer_scale'] = \
    'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/german.numer_scale'
datasets['diabetes_scale'] = \
    'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale'
#-----------------------------------------
# if files don't exist, download them
print "acquiring missing data"
for k,v in datasets.iteritems():
    fname = os.path.join("../data",k)
    if not(os.path.isfile(fname)):
        print "\tdownloading ", k
        print "\turl: ", v
        download(v)
        b = string.split(v,sep='.')
        if b[-1] == 'bz2':
            command = 'bunzip2 '+ k+'.bz2'
            os.system(command)
        shutil.move(k,fname)
    else:
        print "\t",k,"exists."

#-----------------------------------------
# concatenate data sets
# Some data sets have pre-defined train and test sets
# we concatenate them and split them into train, test, and validation sets
print "concatenating data sets"

# checking a9a
fname = "../data/a9a_all"
if not(os.path.isfile(fname)):
	print "\ta9a"
	command = "cat ../data/a9a ../data/a9a.t > ../data/a9a_all"
	os.system(command)

# # checking rcv1
# fname = "../data/rcv1_all"
# if not(os.path.isfile(fname)):
# 	print "\trcv1"
# 	command = "cat ../data/rcv1_test.binary ../data/rcv1_train.binary > ../data/rcv1_all"
# 	os.system(command)

if os.path.isfile('../data/astroph.train') and os.path.isfile('../data/astroph.test'):
	fname = "../data/astro-ph_all"
	if not(os.path.isfile(fname)):
		print "\tastro-ph"
		command = "cat ../data/astroph.train ../data/astroph.test > ../data/astro-ph_all"
		os.system(command)

#-----------------------------------------
# split data

# list of files with all data
data_all = {'../data/a9a_all':'../data/a9a', 
	    '../data/real-sim':'../data/real-sim',
	    # '../data/rcv1_all':'../data/rcv1', 
	    '../data/news20.binary':'../data/news20',
	    '../data/german.numer_scale':'../data/german', 
	    '../data/diabetes_scale':'../data/diabetes'}
if os.path.isfile('../data/astro-ph_all'):
	data_all['../data/astro-ph_all'] = '../data/astro-ph';

print "splitting data"
for k,v in data_all.iteritems():
    #print k,v
    data_split(k,v)


