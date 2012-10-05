import os
import glob
import string

def print_config(fname,config):
    f = open(fname,'w')
    f.write("# Configuration file for erlpboost\n")
    f.write("\n")
    f.write("# Read training data from this file (LIBSVM format)\n")

    trainstr = "train_file = ../data/%s.train\n"%(config['data'])
    f.write(trainstr)
    f.write("\n")

    f.write("# Read testing data from this file (LIBSVM format)\n")
    teststr = "test_file = ../data/%s.test\n"%(config['data'])
    f.write(teststr)
    f.write("\n")

    f.write("# Read validation data from this file (LIBSVM format)\n")
    validstr = "valid_file = ../data/%s.valid\n"%(config['data'])
    f.write(validstr)
    f.write("\n")


    f.write("# Dump all results and output into this file\n")
    if config.has_key('eta'):
        if config.has_key('opt'):
            outstr = "output_file = ../results/%s/%s.%s.%s.%s.%s.%s.%s.%s.output\n"%\
                (config['dir'],config['data'],config['alg'],config['wl'],\
                     config['ref'],config['eps'],config['nuN'],config['eta'],config['opt'])
        else:
            outstr = "output_file = ../results/%s/%s.%s.%s.%s.%s.%s.%s.output\n"%\
                (config['dir'],config['data'],config['alg'],config['wl'],\
                     config['ref'],config['eps'],config['nuN'],config['eta'])
    elif config['alg'] == 'ada':
        outstr = "output_file = ../results/%s/%s.%s.%s.%s.output\n"%\
                 (config['dir'],config['data'],config['alg'],config['wl'],\
                  config['ref'])
    else: 
        outstr = "output_file = ../results/%s/%s.%s.%s.%s.%s.%s.output\n"%\
            (config['dir'],config['data'],config['alg'],config['wl'],\
                 config['ref'],config['eps'],config['nuN'])
    f.write(outstr)
    f.write("\n")


    f.write("# What kind of oracle to use\n")
    f.write("# Possible values are rawdata, decisionstump, or svm\n")
    if config['wl']=='ds':
        wl = "decisionstump"
    elif config['wl']=='rd':
        wl = "rawdata"
    else:
        wl = "svm"
    oraclestr = "oracle_type = %s\n"%(wl)
    f.write(oraclestr)
    f.write("\n")


    f.write("# Should the weak learner set the reflexive flag?\n")
    if config['ref']=='t':
        ref="true"
    else:
        ref="false"
    refstr = "reflexive = %s\n"%(ref)
    f.write(refstr)
    f.write("\n")

    f.write("# Maximum number of iterations of boosting\n")
    if config.has_key('max_iter'):
        tmpstr = "max_iter = %s\n"%(config['max_iter'])
        f.write(tmpstr)
    else:
        f.write("max_iter = 1000\n")
    f.write("\n")

    if config['alg'] != 'ada':
        f.write("# Epsilon tolerance\n")
        if config.has_key('eps'):
            epstr = "eps = 0.%s\n"%(config['eps'])
            f.write(epstr)
        else:
            f.write("eps = 0.01\n")
        f.write("\n")
    
    if config.has_key('eta'):
        f.write("# Regularization Parameter\n")
        etastr = "eta = %s.0\n"%(config['eta'])
        f.write(etastr)
        f.write("\n")

    if config['alg'] != 'ada':
        f.write("# nu for softening. Actually 1/nu is used in the code\n")
        if config.has_key('nu'):
            nustr = "nu = %s\n"%(config['nu'])
            f.write(nustr)
        else:
            f.write("nu = 1.0\n")
        f.write("\n")
    
    
    if config['alg'] == 'bin':
        f.write("# binary boost or normal ERLPBoost\n")
        f.write("binary = true\n")
        f.write("\n")
    elif config['alg'] == 'erlp':
        f.write("# binary boost or normal ERLPBoost\n")
        f.write("binary = false\n")
        f.write("\n")

    if config.has_key('booster_type'):
        f.write("# type of boosting algorithm\n")
        f.write("# choices are ERLPBoost, AdaBoost, Corrective, and LPBoost\n")
        booststr = "booster_type = %s\n"%(config['booster_type'])
        f.write(booststr)
    f.write("\n")

    if config['alg'] == 'bin' or config['alg'] == 'erlp':
        f.write("# What kind of optimizer to use\n")
        f.write("# Possible values are tao, hz, pg and cd\n")
        if config.has_key('opt'):
            optstr = "optimizer_type = %s\n"%(config['opt'])
            f.write(optstr)
        else:
            f.write("optimizer_type = tao\n")




#-------------------------------------
# Methods to print for each algorithm
#-------------------------------------

def print_lp(wl,eps,mode):
    data = ['news20','real-sim','astro-ph','a9a','rcv1','german','diabetes']
    #data = ['german','diabetes']
    #ref = ['t','f']
    ref = ['t']
    
    config = {}
    config['wl'] = wl
    config['dir'] = 'lp'
    config['eps'] = eps
    config['nuN'] = 'nc'
    config['nu'] = '1.0'
    config['alg'] = 'lp'
    config['booster_type'] = 'LPBoost'
    
    if mode==0:
        for r in ref:
            for d in data:
                config['data'] = d
                config['ref'] = r
                outfile = "../config/lp/%s.%s.%s.%s.%s.%s.conf"%\
                    (config['data'],config['alg'],config['wl'],config['ref'],\
                         config['eps'],config['nuN'])
                print outfile
                print_config(outfile,config)
    if mode==1:
        nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        NuN = ['1','2','3','4','5','6','7','8','9']
        # number of data points in training set
        Ns = [15960,43385,56913, 29305,418584,600,460]
        #Ns = [600,460]
        i = -1;
        for r in ref:
            for n in xrange(len(nus)):
                for i in xrange(len(data)):
                    config['data'] = data[i]
                    config['nuN'] = NuN[n]
                    config['nu'] = nus[n] * Ns[i]
                    config['ref'] = r
                    outfile = "../config/lp/%s.%s.%s.%s.%s.%s.conf"%\
                        (config['data'],config['alg'],config['wl'],config['ref'],\
                             config['eps'],config['nuN'])
                    print outfile #, i, Ns[i], config['nuN'], config['nu']
                    print_config(outfile,config)


def print_ada(wl,max_iter = '50000'):
    data = ['news20','real-sim','astro-ph','a9a','rcv1','german','diabetes']
    #data = ['german','diabetes']
    #ref = ['t','f']
    ref = ['t']
    config = {}
    config['dir'] = 'ada'
    config['ref'] = ref
    config['booster_type'] = 'AdaBoost'
    config['alg'] = 'ada'
    config['wl'] = wl
    config['max_iter'] = max_iter
    
    for r in ref:
        for d in data:
                config['data'] = d
                config['ref'] = r
                outfile = "../config/ada/%s.%s.%s.%s.conf"%\
                    (config['data'],config['alg'],config['wl'],config['ref'])
                print outfile
                print_config(outfile,config)

def print_corrective(wl,eps,max_iter,mode):
    """
    mode 0: default eta, no capping
    mode 1: vary eta, no capping
    mode 2: default eta, capping
    """
    data = ['news20','real-sim','astro-ph','a9a','rcv1','german','diabetes']
    ref = ['t']
    eta = ['1','2','5','10','20','50','100','200','500','1000','2000','3000']

    config = {}
    config['dir'] = 'corr'
    config['eps'] = eps
    config['nuN'] = 'nc'
    config['alg'] = 'corr'
    config['wl'] = wl
    config['max_iter'] = max_iter
    config['booster_type'] = 'Corrective'

    if mode==0:
        for d in data:
                for r in ref:
                    config['data'] = d
                    config['ref'] = r
                    outfile = "../config/corr/%s.%s.%s.%s.%s.%s.conf"%\
                        (config['data'],config['alg'],config['wl'],config['ref'],\
                             config['eps'],config['nuN'])
                    print outfile
                    print_config(outfile,config)
    elif mode==1:
        for d in data:
            for r in ref:
                for e in eta:
                    config['data'] = d
                    config['ref'] = r
                    config['eta'] = e
                    outfile = "../config/corr/%s.%s.%s.%s.%s.%s.%s.conf"%\
                        (config['data'],config['alg'],config['wl'],config['ref'],\
                             config['eps'],config['nuN'],config['eta'])
                    print outfile
                    print_config(outfile,config)
                    
    elif mode==2:
        nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        NuN = ['1','2','3','4','5','6','7','8','9']
        # number of data points in training set
        Ns = [15960,43385,56913, 29305,418584,600,460]
        #Ns = [600,460]
        i = -1;
        for r in ref:
            for n in xrange(len(nus)):
                for i in xrange(len(data)):
                    config['data'] = data[i]
                    config['nuN'] = NuN[n]
                    config['nu'] = nus[n] * Ns[i]
                    config['ref'] = r
                    outfile = "../config/corr/%s.%s.%s.%s.%s.%s.conf"%\
                        (config['data'],config['alg'],config['wl'],config['ref'],\
                             config['eps'],config['nuN'])
                    print outfile 
                    print_config(outfile,config)
 

def print_erlp(wl,eps,binary,mode):
    """
    mode 0: default eta, no capping
    mode 1: vary eta, no capping
    mode 2: default eta, capping
    """
    data = ['news20','real-sim','astro-ph','a9a','rcv1','german','diabetes']
    #data = ['news20','real-sim','astro-ph','a9a','german','diabetes']
    #data = ['german','diabetes']

    #ref = ['t','f']
    ref = ['t']
    eta = ['1','2','5','10','20','50','100','200','500','1000','2000','3000']
    config = {}
    if binary:
        config['dir'] = 'bin'
        config['alg'] = 'bin'
    else:
        config['dir'] = 'erlp'
        config['alg'] = 'erlp'

    config['eps'] = eps
    config['nuN'] = 'nc'
    config['wl'] = wl
    config['max_iter'] = 1000

    if mode==0:
        for d in data:
            for r in ref:
                config['data'] = d
                config['ref'] = r
                outfile = "../config/%s/%s.%s.%s.%s.%s.%s.conf"%\
                    (config['alg'],config['data'],config['alg'],\
                         config['wl'],config['ref'],\
                         config['eps'],config['nuN'])
                print outfile
                print_config(outfile,config)
    if mode==1:
        for d in data:
            for e in eta:	
                for r in ref:
                    config['eta'] = e
                    config['data'] = d
                    config['ref'] = r
                    outfile = "../config/%s/%s.%s.%s.%s.%s.%s.%s.conf"%\
                        (config['alg'],config['data'],config['alg'],\
                             config['wl'],config['ref'],config['eps'],\
                             config['nuN'],config['eta'])
                    print outfile
                    print_config(outfile,config)
    if mode==2:
        nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        NuN = ['1','2','3','4','5','6','7','8','9']
        # number of data points in training set
        Ns = [15960,43385,56913, 29305,418584,600,460]
        #Ns = [600,460]
        i = -1;
        for r in ref:
            for n in xrange(len(nus)):
                for i in xrange(len(data)):
                    config['data'] = data[i]
                    config['nuN'] = NuN[n]
                    config['nu'] = nus[n] * Ns[i]
                    config['ref'] = r
                    outfile = "../config/%s/%s.%s.%s.%s.%s.%s.conf"%\
                        (config['alg'],config['data'],config['alg'],\
                             config['wl'],config['ref'],\
                             config['eps'],config['nuN'])
                    print outfile #, i, Ns[i], config['nuN'], config['nu']
                    print_config(outfile,config)
 

def print_opt(wl,eps,binary,opt):
    #data = ['news20','real-sim','astro-ph','a9a','rcv1','german','diabetes']
    data = ['real-sim']
    ref = ['t']
    eta = ['1','2','5','10','20','50','100','200','500','1000','2000','3000']
    #opt = ['tao','hz','pg']


    config = {}
    if binary:
        config['alg'] = 'bin'
    else:
        config['alg'] = 'erlp'
    config['opt'] = opt
    config['dir'] = 'opt'
    config['eps'] = eps
    config['nuN'] = 'nc'
    config['wl'] = wl
    config['max_iter'] = 1000


    for d in data:
        for e in eta:	
            for r in ref:
                config['eta'] = e
                config['data'] = d
                config['ref'] = r
                outfile = "../config/%s/%s.%s.%s.%s.%s.%s.%s.%s.conf"%\
                    (config['dir'],config['data'],config['alg'],\
                         config['wl'],config['ref'],config['eps'],\
                         config['nuN'],config['eta'],config['opt'])
                print outfile
                print_config(outfile,config)


if __name__ == "__main__":



#     # adaboost
#     print_ada('svm','5000')
#     print_ada('ds','5000')
#     print_ada('rd','5000')

#     # lpboost
#     print_lp('svm','05',0)
#     print_lp('svm','05',1)



    # print the corrective config files
    for i in xrange(1,3):
        print_corrective('ds','01','10000',i)
        print_corrective('rd','01','10000',i)
        #print_corrective('svm','05','10000',i)

#     # print the erlp config files
#     for i in xrange(1,3):
#         print_erlp('rd','001',False,i)
#         print_erlp('ds','001',False,i)
#         #print_erlp('svm','001',False,i)


#     # print the binary erlp config files
#     for i in xrange(1,3):
#         print_erlp('rd','001',True,i)
#         print_erlp('ds','001',True,i)
#         #print_erlp('svm','001',True,i)

    #print_opt('ds','05',False,'hz')
#     opt = ['tao','hz','pg','cd']
#     for o in opt:
#         print_opt('ds','01',False,o)
